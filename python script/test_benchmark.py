import os
import utils
import argparse
import gdown
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
from utils import map_model_to_url, map_benchmark_to_folder

BENCHMARK_DIR = "dataset/benchmark/"
PREDICTION_DIR = "predictions/"
FI_DIR = BENCHMARK_DIR + "FI/"

def predict_model(model_name, dataset_name, vit_model):
  '''Returns the accuracy obtained on the benchmark passed in input with the model corresponding to the path given by 'model_name'.'''

  complete_model = utils.get_model_architecture(vit_model)  # Get the model structure

  test_annot = pd.read_csv(map_benchmark_to_folder[dataset_name] + dataset_name)  # get the dataframe with the gold labels
  benchmark = utils.get_dataset(test_annot, dataset_name, vit_model)

  model_weights_path = "models/" + model_name + ".h5"

  if not os.path.exists(model_weights_path):
    url = map_model_to_url[model_name]
    print("Downloading file...")
    gdown.download(url, model_weights_path)
    print("Done!")
  else:
    print("Model", model_name, "already downloaded!")

  complete_model.load_weights(model_weights_path)  # load the saved weights

  if vit_model == "b32":
    complete_model = tf.keras.Sequential([
      layers.Rescaling(1.0 / 255),
      complete_model
    ], name="complete_model")

  predictions = complete_model.predict(benchmark)  # predict the labels

  pred_df = pd.DataFrame(predictions, columns=["NEG", "NEU", "POS"])
  pred_df.to_csv(PREDICTION_DIR + model_name + "_" + dataset_name.split(".")[0] + ".csv", index=None)

  bin_predictions = np.delete(predictions, 1, 1)  # remove the Neutral prediction, since the benchmark is a binary classification problem
  pred_labels = np.argmax(bin_predictions, axis=1).tolist()

  if dataset_name.startswith("FI"):
    gold_labels = test_annot['class'].apply(change_labels).apply(int).tolist()
  else:
    gold_labels = test_annot['class'].apply(int).tolist()

  curr_accuracy = accuracy_score(pred_labels, gold_labels)  # compute the accuracy

  return curr_accuracy


def predict_5_fold(dataset_dir, dataset_name):
  accuracies = []
  dataset_to_test = pd.read_csv(dataset_dir + "/" + dataset_name)  # get the dataframe with the gold labels

  if dataset_dir.split("/")[-1] == "Twitter Testing Dataset I":
    base_dir = os.path.join(dataset_dir) + "/models/" + dataset_name.split(".")[0] + "/"
  else:
    base_dir = os.path.join(dataset_dir) + "/models/"

  labels = dataset_to_test[['class']]

  skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

  for i, (_, test_index) in enumerate(skf.split(np.zeros(dataset_to_test.shape[0]), labels)):

    test_annot = dataset_to_test.loc[test_index]
    model = "fold_" + str(i) + ".h5"
    complete_model = utils.get_5_fold_architecture(base_dir + model)  # Get the model structure

    benchmark = utils.get_dataset(test_annot, "l16", dataset_name)

    predictions = complete_model.predict(benchmark)  # predict the labels

    pred_labels = np.argmax(predictions, axis=1).tolist()
    gold_labels = test_annot['class'].apply(int).tolist()
    gold_labels = [elem if elem == 0 else 2 for elem in gold_labels]

    acc = accuracy_score(pred_labels, gold_labels)
    accuracies.append(acc)  # compute the accuracy
    print("Accuracy fold", i, ":", acc)

  aggregated_accuracy = np.mean(accuracies)
  std = np.std(accuracies)

  return aggregated_accuracy, std


def predict_FI_split():
  accuracies = []

  for split in range(1, 5):

    test_annot = pd.read_csv(FI_DIR + "split_" + str(split) + "/FI_test.csv")  # get the dataframe with the gold labels
    benchmark = utils.get_dataset(test_annot, "l16", "FI_test.csv")
    complete_model = utils.get_model_architecture("l16")  # Get the model structure

    complete_model.load_weights(FI_DIR + "models/model_" + str(split) + ".h5")  # load the saved weights

    predictions = complete_model.predict(benchmark)  # predict the labels

    bin_predictions = np.delete(predictions, 1, 1)  # remove the Neutral prediction, since the benchmark is a binary classification problem
    pred_labels = np.argmax(bin_predictions, axis=1).tolist()

    gold_labels = test_annot['class'].apply(change_labels).apply(int).tolist()

    acc = accuracy_score(pred_labels, gold_labels)
    print("Accuracy fold", str(split), ":", acc)
    accuracies.append(acc)  # compute the accuracy

  aggregated_accuracy = np.mean(accuracies)
  std = np.std(accuracies)

  return aggregated_accuracy, std


def main():
  parser = argparse.ArgumentParser(description='Testing the model with the B-T4SA test set.')
  parser.add_argument("--model", "-m", type=str, help="Model to test", required=False)
  parser.add_argument("--dir", type=str, help="Model to test", required=False)
  parser.add_argument("--benchmark", "-b", type=str, help="Benchmark to test", required=False)
  parser.add_argument("--five_fold", type=bool, default=False, help="Five fold validation")
  parser.add_argument("--FI_split", type=bool, default=False, help="FI validation")

  args = parser.parse_args()

  benchmark = args.benchmark
  model = args.model

  vit_type = utils.get_vit_type(model)

  if args.FI_split:
    acc, std = predict_FI_split()
    print("Mean accuracy:", acc)
    print("Standard deviation:", std)
  elif args.five_fold:
    acc, std = predict_5_fold("dataset/benchmark/" + args.dir, dataset_name=benchmark)
    print("Mean accuracy:", acc)
    print("Standard deviation:", std)
  else:
    print(predict_model(model, dataset_name=benchmark + ".csv", vit_model=vit_type))



if __name__ == "__main__":
  main()
  exit(0)
