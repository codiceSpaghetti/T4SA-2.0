import os
import gdown
import utils
import argparse
import numpy as np
import pandas as pd
import random as rn
import tensorflow as tf
import tensorflow_addons as tfa

from vit_keras import vit
from tensorflow import keras
from utils import  map_model_to_url
from IPython.display import display
from tensorflow.keras import layers
from keras.callbacks import Callback
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight


BENCHMARK_DIR = "dataset/benchmark/"
FI_DIR = BENCHMARK_DIR + "FI/"
PREDICTION_DIR = "predictions/"

IMAGE_SIZE = 384


class WeightsSaver(Callback):
  def __init__(self, N, split, batch_size):
    self.N = N
    self.batch = 0
    self.epoch = 0
    self.best_val_accuracy = 0
    self.split = split

    self.val_annot = pd.read_csv(FI_DIR + "split_" + str(split) + "/FI_val.csv")
    self.valid_dataset = build_data_pipeline(self.val_annot).batch(batch_size).prefetch(tf.data.AUTOTUNE)

  def on_batch_end(self, batch, logs={}):
    if self.batch % self.N == 0:
      curr_accuracy = compute_accuracy(self.model, self.valid_dataset, self.val_annot, self.split)
      if curr_accuracy > self.best_val_accuracy:
        self.best_val_accuracy = curr_accuracy
        name = FI_DIR + 'models/model_' + str(self.split) + '.h5'
        self.model.save(name)
        print(f"Saving model after {self.batch} batches with accuracy", curr_accuracy)
    self.batch += 1

  def on_epoch_end(self, epoch, logs={}):
    curr_accuracy = compute_accuracy(self.model, self.valid_dataset, self.val_annot, self.split)
    if curr_accuracy > self.best_val_accuracy:
      self.best_val_accuracy = curr_accuracy
      name = FI_DIR + 'models/model_' + str(self.split) + '.h5'
      self.model.save(name)
      print(f"Saving model after {self.epoch} epoch with accuracy", curr_accuracy)

    self.epoch += 1


def compute_accuracy(model, dataset, test_annot, split):
  predictions = model.predict(dataset)

  pred_df = pd.DataFrame(predictions, columns=["NEG", "NEU", "POS"])
  pred_df.to_csv(PREDICTION_DIR + "fine_tune_FI_" + str(split) + ".csv", index=None)

  bin_predictions = np.delete(predictions, 1, 1)  # remove the Neutral prediction, since the
                                                  # benchmark is a binary classification problem
  pred_labels = np.argmax(bin_predictions, axis=1).tolist()

  gold_labels = test_annot['class'].apply(int).tolist()
  gold_labels = [elem if elem == 0 else 1 for elem in gold_labels]

  accuracy = accuracy_score(pred_labels, gold_labels)  # compute the accuracy
  return accuracy


def load_image_tf(path):
  image_data = tf.io.read_file(FI_DIR + "images/" + path)  # read image file
  image = tf.image.decode_image(image_data, channels=3, expand_animations=False)  # decode image data as RGB (do not load whole animations, i.e., GIFs)
  image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))  # resize
  image = vit.preprocess_inputs(image)

  return image


def build_data_pipeline(annot):
  image_paths = annot.path.map(str)  # Path -> str
  labels = annot["class"].map(float)  # integer labels of categories

  data = tf.data.Dataset.from_tensor_slices((image_paths, labels))
  data = data.map(
    lambda x, y: (load_image_tf(x), y),  # path -> image, keep y unaltered
    num_parallel_calls=tf.data.AUTOTUNE,  # load in parallel
    deterministic=True  # keep the order (we will shuffle afterward if needed)
  )
  return data


def main():
  os.environ["PYTHONHASHSEED"] = "0"  # necessary for reproducible results of certain Python hash-based operations.
  rn.seed(14)  # necessary for starting core Python generated random numbers in a well-defined state.
  np.random.seed(42)  # necessary for starting Numpy generated random numbers in a well-defined initial state.
  tf.random.set_seed(42)  # necessary to random number generation in TensorFlow have a well-defined initial state.

  parser = argparse.ArgumentParser(description='Fine tune the boosted model on the FI dataset.')
  parser.add_argument("--batch_size", "-batch", type=int, default=32, help="Batch size of training")
  parser.add_argument("--epochs", type=int, default=5, help="Number of epochs of training")
  parser.add_argument("--split", type=int, default=5, help="Number of split of the test set")

  args = parser.parse_args()

  BATCH_SIZE = args.batch_size
  EPOCHS = args.epochs
  NUM_SPLIT = args.split

  test_accuracies = []

  for split in range(1, NUM_SPLIT+1):
    print("Reading csv files with training and validation set for split", str(split), "...")
    train_annot = pd.read_csv(FI_DIR + "split_" + str(split) + "/FI_train.csv")
    test_annot = pd.read_csv(FI_DIR + "split_" + str(split) + "/FI_test.csv")

    train_dataset = build_data_pipeline(train_annot).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = build_data_pipeline(test_annot).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model_name = "boosted_model"
    model_weights_path = "models/" + model_name + ".h5"

    if not os.path.exists(model_weights_path):
      url = map_model_to_url[model_name]
      print("Downloading file...")
      gdown.download(url, model_weights_path)
      print("Done!")
    else:
      print("Model", model_name,"already downloaded!")

    model = tf.keras.models.load_model(model_weights_path)

    learning_rate = 1e-4
    optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    model.trainable = True
    model.summary()

    print("Training the model fully unfreezed...")

    model.fit(x=train_dataset,
              epochs=EPOCHS,
              callbacks=[WeightsSaver(200, split, BATCH_SIZE)],
              verbose=1)

    model.load_weights(FI_DIR + 'models/model_' + str(split) + '.h5')

    curr_accuracy = compute_accuracy(model, test_dataset, test_annot, split)
    test_accuracies.append(curr_accuracy)

  print("Accuracy obtained:", test_accuracies)
  aggregated_result = sum(test_accuracies) / len(test_accuracies)
  print("Aggregated accuracy:", aggregated_result)
  print("Standard deviation:", np.std(test_accuracies))


if __name__ == "__main__":
  main()
