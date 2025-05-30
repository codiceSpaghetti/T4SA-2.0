# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import pandas as pd
import random as rn
import tensorflow as tf
import tensorflow_addons as tfa

from vit_keras import vit
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL_DIR = "/workdir/data/final_models/"
BENCHMARK_DIR = "/workdir/data/benchmark/"

TWITTER_1_DIR = BENCHMARK_DIR + "Twitter1269/"
TWITTER_2_DIR = BENCHMARK_DIR + "AMT_Twitter/"
EMOTION_ROI_DIR = BENCHMARK_DIR + "EmotionROI/"
FI_DIR = BENCHMARK_DIR + "emotion_dataset/"

IMAGE_SIZE = 384

# Utilities
map_dataset_to_folder = {TWITTER_1_DIR + "3agree.csv": TWITTER_1_DIR + "images/",
                         TWITTER_1_DIR + "4agree.csv": TWITTER_1_DIR + "images/",
                         TWITTER_1_DIR + "5agree.csv": TWITTER_1_DIR + "images/",
                         TWITTER_2_DIR + "twitter_testing_2.csv": TWITTER_2_DIR + "images/",
                         FI_DIR + "FI.csv": FI_DIR + "images/",
                         EMOTION_ROI_DIR + "emotion_ROI_test.csv": EMOTION_ROI_DIR + "images/",
                         EMOTION_ROI_DIR + "emotion_ROI_train.csv": EMOTION_ROI_DIR + "images/",
                         EMOTION_ROI_DIR + "emotion_ROI_complete.csv": EMOTION_ROI_DIR + "images/"}


def load_image_tf(path, dataset_name):
  '''Decodes the image specified by the path in input and applies some preprocessing to it.'''
  image_data = tf.io.read_file(map_dataset_to_folder[dataset_name] + path)  # read image file
  image = tf.image.decode_image(image_data, channels=3, expand_animations=False)  # decode image data as RGB (do not load whole animations, i.e., GIFs)
  image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))  # resize
  image = vit.preprocess_inputs(image)

  return image


def get_dataset(annot, dataset_name=TWITTER_1_DIR + "3agree.csv", batch_size=32):
  '''Returns a tf.data.Dataset that maps image and labels taken from the dataframe in input.'''
  x = annot['path'].to_list()
  labels = annot['class'].apply(int).to_list()
  y = [elem if elem == 0 else 2 for elem in labels]

  # Buld a tensorflow dataset
  data = tf.data.Dataset.from_tensor_slices((x, y))

  # Map the dataset with the preprocessing function
  data = data.map(
    lambda x, y: (load_image_tf(x, dataset_name), y),  # path -> image, keep y unaltered
    num_parallel_calls=tf.data.AUTOTUNE,  # load in parallel
    deterministic=True  # keep the order
  ).batch(batch_size)

  return data


def get_model_architecture(model_name):
  '''Returns a tf.data.Dataset that maps image and labels taken from the dataframe in input.'''
  # Load the base ViT model
  vit_model = vit.vit_l16(
      image_size=IMAGE_SIZE,
      pretrained=True,
      include_top=False,
      pretrained_top=False
  )

  vit_model.trainable = False

  # Add the classification head
  model = tf.keras.Sequential([
    vit_model,
    layers.Dense(32, activation=tfa.activations.gelu),
    layers.Dense(3, activation='softmax')
  ],
    name='vision_transformer')

  model.load_weights(model_name)  # load the saved weights

  mask = tf.convert_to_tensor([1.0, 0.0000001, 1.0], dtype=tf.float32)
  mask = tf.expand_dims(tf.cast(mask, dtype=tf.float32), axis=0)

  inputs = model.inputs
  x = model(inputs)
  outputs = tf.keras.layers.Multiply()([x, mask])

  model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

  return model


def predict_model_on_small_dataset(model_name, dataset_dir, dataset_name, K, batch_size, epochs):
  test_accuracies = []

  dataset_to_test = pd.read_csv(dataset_dir + dataset_name)  # get the dataframe with the gold labels
  labels = dataset_to_test[['class']]

  skf = StratifiedKFold(n_splits=K, random_state=42, shuffle=True)

  for i, (train_index, test_index) in enumerate(skf.split(np.zeros(dataset_to_test.shape[0]), labels)):

    train_annot = dataset_to_test.loc[train_index]
    test_annot = dataset_to_test.loc[test_index]

    test_annot['class'] = test_annot['class'].apply(int)

    train_dataset = get_dataset(train_annot, dataset_dir + dataset_name, batch_size=batch_size)
    test_dataset = get_dataset(test_annot, dataset_dir + dataset_name, batch_size=batch_size)

    new_fold_model = get_model_architecture(model_name)  # Get the model structure

    learning_rate = 1e-4
    optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)

    new_fold_model.compile(optimizer=optimizer,
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])

    new_fold_model.trainable = True

    new_fold_model.fit(x=train_dataset,
                       epochs=epochs,
                       verbose=1)

    new_fold_model.save(dataset_dir + "models/" + dataset_name.split(".")[0] + "/" + dataset_name.split(".")[0] + "_fold_" + str(i) + ".h5")
    new_fold_model.load_weights(dataset_dir + "models/" + dataset_name.split(".")[0] + "/" + dataset_name.split(".")[0] + "_fold_" + str(i) + ".h5")

    predictions = new_fold_model.predict(test_dataset)  # predict the labels

    pred_labels = np.argmax(predictions, axis=1).tolist()
    gold_labels = test_annot['class'].apply(int).tolist()
    gold_labels = [elem if elem == 0 else 2 for elem in gold_labels]

    curr_accuracy = accuracy_score(pred_labels, gold_labels)  # compute the accuracy

    print("Accuracy fold", i, ":", curr_accuracy)
    test_accuracies.append(curr_accuracy)

  print("Accuracy obtained for", dataset_name, ":", test_accuracies)
  aggregated_result = np.mean(test_accuracies)
  print("Aggregated accuracy:",  aggregated_result)
  print("Standard deviation:", np.std(test_accuracies))

  return aggregated_result


def main():
  os.environ["PYTHONHASHSEED"] = "0"  # necessary for reproducible results of certain Python hash-based operations.

  rn.seed(14)  # necessary for starting core Python generated random numbers in a well-defined state.

  np.random.seed(42)  # necessary for starting Numpy generated random numbers in a well-defined initial state.

  tf.random.set_seed(42)  # necessary to random number generation in TensorFlow have a well-defined initial state.

  print("Reading csv files with training and validation set...")

  parser = argparse.ArgumentParser(description='Fine tune the model on small-scale datasets using K-Fold cross validation.')
  parser.add_argument("--model", "-m", type=str, default=MODEL_DIR + "boosted_model.h5", help="Name of the model")
  parser.add_argument("--dir", type=str, help="Directory with the benchmark images")
  parser.add_argument("--benchmark", "-b",  type=str, help="Benchmark to test")
  parser.add_argument("--batch_size", "-batch", type=int, default=32, help="Batch size of training")
  parser.add_argument("--K", type=int, default=5, help="Number of folds for cross validation")
  parser.add_argument("--epochs", "-e", type=int, default=12, help="Number of epochs of training")

  args = parser.parse_args()

  predict_model_on_small_dataset(args.model, dataset_dir=args.dir, dataset_name=args.benchmark, batch_size=args.batch_size, K=args.K, epochs=args.epochs)
  #predict_model_on_small_dataset(MODEL_DIR + "boosted_model.h5", dataset_dir=TWITTER_1_DIR, dataset_name="3agree.csv", vit_model="l16")


if __name__ == "__main__":
  main()
