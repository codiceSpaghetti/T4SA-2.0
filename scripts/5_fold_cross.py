import os
import utils
import gdown
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
from utils import image_size_mapper, map_dataset_to_folder, map_model_to_url, map_benchmark_to_folder

PREDICTION_DIR = "predictions/"

def get_model_architecture(model_name):
  '''Returns a tf.data.Dataset that maps image and labels taken from the dataframe in input.'''
  # Load the base ViT model
  vit_model = vit.vit_l16(
      image_size=384,
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

  model_weights_path = "models/" + model_name + ".h5"

  if not os.path.exists(model_weights_path):
    url = map_model_to_url[model_name]
    print("Downloading file...")
    gdown.download(url, model_weights_path)
    print("Done!")
  else:
    print("Model", model_name, "already downloaded!")

  skf = StratifiedKFold(n_splits=K, random_state=42, shuffle=True)

  for i, (train_index, test_index) in enumerate(skf.split(np.zeros(dataset_to_test.shape[0]), labels)):

    train_annot = dataset_to_test.loc[train_index]
    test_annot = dataset_to_test.loc[test_index]

    test_annot['class'] = test_annot['class'].apply(int)

    train_dataset = utils.get_dataset(train_annot, dataset_name, batch_size=batch_size)
    test_dataset = utils.get_dataset(test_annot, dataset_name, batch_size=batch_size)

    new_fold_model = get_model_architecture(model_weights_path)  # Get the model structure

    learning_rate = 1e-4
    optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)

    new_fold_model.compile(optimizer=optimizer,
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])

    new_fold_model.trainable = True

    new_fold_model.fit(x=train_dataset,
                       epochs=epochs,
                       verbose=1)

    new_fold_model.save(dataset_dir + "models/" + dataset_name.split(".")[0] + "/fold_" + str(i) + ".h5")
    new_fold_model.load_weights(dataset_dir + "models/" + dataset_name.split(".")[0] + "/fold_" + str(i) + ".h5")

    predictions = new_fold_model.predict(test_dataset)  # predict the labels

    pred_df = pd.DataFrame(predictions, columns=["NEG", "NEU", "POS"])
    pred_df.to_csv(PREDICTION_DIR + "fine_tune_" + dataset_name.split(".")[0] + "_" + str(i) + ".csv", index=None)

    pred_labels = np.argmax(predictions, axis=1).tolist()
    gold_labels = test_annot['class'].apply(int).tolist()
    gold_labels = [elem if elem == 0 else 2 for elem in gold_labels]

    curr_accuracy = accuracy_score(pred_labels, gold_labels)  # compute the accuracy

    print("Accuracy fold", str(i), ":", curr_accuracy)
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

  parser = argparse.ArgumentParser(description='Fine tune the model on small-scale datasets using K-Fold cross validation.')
  parser.add_argument("--model", "-m", type=str, default="boosted_model", help="Name of the model")
  #parser.add_argument("--dir", type=str, help="Directory with the benchmark images")
  parser.add_argument("--benchmark", "-b",  type=str, help="Benchmark to test")
  parser.add_argument("--batch_size", "-batch", type=int, default=32, help="Batch size of training")
  parser.add_argument("--K", type=int, default=5, help="Number of folds for cross validation")
  parser.add_argument("--epochs", "-e", type=int, default=12, help="Number of epochs of training")

  args = parser.parse_args()
  BENCHMARK = args.benchmark + ".csv"

  predict_model_on_small_dataset(args.model, dataset_dir=map_benchmark_to_folder[BENCHMARK], dataset_name=BENCHMARK, batch_size=args.batch_size, K=args.K, epochs=args.epochs)


if __name__ == "__main__":
  main()
