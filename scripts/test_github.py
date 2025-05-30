import os
import argparse
import numpy as np
import pandas as pd
import random as rn
import tensorflow as tf
import tensorflow_addons as tfa

from vit_keras import vit
from tensorflow import keras
from IPython.display import display
from tensorflow.keras import layers
from keras.callbacks import Callback
from sklearn.metrics import accuracy_score

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL_DIR = "/workdir/data/final_models/"
DF_DIR = "/workdir/data/test_sets/"
DATASET_DIR = "/workdir/data/crawling_backup/"
PREDICTION_DIR = "/workdir/data/predictions/test_set/"

#MODEL_DIR = "/workdir/data/final_models/"
#DF_DIR = "/workdir/data/test_sets/"
#DATASET_DIR = "/workdir/data/images/"
#PREDICTION_DIR = "/workdir/data/predictions/test_set/"


def get_generator(test_annot, BATCH_SIZE, IMAGE_SIZE):
  datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

  test_gen = datagen.flow_from_dataframe(dataframe=test_annot,
                                         directory=DATASET_DIR,
                                         x_col='path',
                                         y_col='class',
                                         batch_size=BATCH_SIZE,
                                         seed=1,
                                         color_mode='rgb',
                                         shuffle=False,
                                         class_mode='categorical',
                                         target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                         validate_filenames=False)

  return test_gen


def load_image_tf(path, image_size):
  image_data = tf.io.read_file(DATASET_DIR + path)  # read image file
  image = tf.image.decode_image(image_data, channels=3, expand_animations=False)  # decode image data as RGB (do not load whole animations, i.e., GIFs)
  image = tf.image.resize(image, (image_size, image_size))  # resize
  image = vit.preprocess_inputs(image)

  return image


def build_data_pipeline(annot, image_size):
  image_paths = annot.path.map(str)  # Path -> str
  labels = annot["class"].map(float)  # integer labels of categories

  lb = LabelBinarizer()
  labels = lb.fit_transform(labels)

  data = tf.data.Dataset.from_tensor_slices((image_paths, labels))
  data = data.map(
    lambda x, y: (load_image_tf(x, image_size), y),  # path -> image, keep y unaltered
    num_parallel_calls=tf.data.AUTOTUNE,  # load in parallel
    deterministic=True  # keep the order (we will shuffle afterward if needed)
  )
  return data


def get_dataset(test_annot, MODEL_NAME, BATCH_SIZE, IMAGE_SIZE):
  if MODEL_NAME in ['B-T4SA_1.0', 'B-T4SA_1.0_upd', 'B-T4SA_1.0_upd_filt', 'bal_flat_T4SA2.0', 'unb_T4SA2.0', 'bal_T4SA2.0', 'merged_T4SA']:
    return get_generator(test_annot, MODEL_NAME, BATCH_SIZE, IMAGE_SIZE)
  else:
    return build_data_pipeline(test_annot, IMAGE_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def main():
  os.environ["PYTHONHASHSEED"] = "0"  # necessary for reproducible results of certain Python hash-based operations.
  rn.seed(14)  # necessary for starting core Python generated random numbers in a well-defined state.
  np.random.seed(42)  # necessary for starting Numpy generated random numbers in a well-defined initial state.
  tf.random.set_seed(42)  # necessary to random number generation in TensorFlow have a well-defined initial state.

  possible_models = ['B-T4SA_1.0', 'B-T4SA_1.0_upd', 'B-T4SA_1.0_upd_filt', 'bal_flat_T4SA2.0', 'unb_T4SA2.0', 'bal_T4SA2.0', 'merged_T4SA', 'boosted_model']

  parser = argparse.ArgumentParser(description='Testing the model with the B-T4SA test set.')
  parser.add_argument("--model", type=str, choices=possible_models, help="The model to test")
  parser.add_argument("--batch_size", "-batch", type=int, default=256, help="Batch size of testing")
  parser.add_argument("--image_size", type=int, default=384, help="Size of images in input to the model")

  args = parser.parse_args()

  MODEL_NAME = args.model
  BATCH_SIZE = args.batch_size
  IMAGE_SIZE = args.image_size

  print("Reading csv test file...")
  test_annot = pd.read_csv(DF_DIR + MODEL_NAME + "_test.csv")
  display((test_annot))

  test_dataset = get_dataset(MODEL_NAME, BATCH_SIZE, IMAGE_SIZE)

  print(f"Loading the model '{MODEL_NAME}'")
  model = tf.keras.models.load_model(MODEL_DIR + MODEL_NAME + ".h5")
  model = tf.keras.models.Model(inputs=model.inputs, outputs=model.get_layer('dense_1').output)

  learning_rate = 1e-4
  optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)

  model.compile(optimizer=optimizer,
                loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
                metrics=['accuracy'])

  print("Predicting with the model, it can take a while...")
  predictions = model.predict(test_dataset)

  pred_labels = np.argmax(predictions, axis=1).tolist()
  gold_labels = test_annot['class'].apply(int).tolist()

  accuracy = accuracy_score(pred_labels, gold_labels)  # compute the accuracy
  print(f"Model accuracy: {accuracy}")

  with open(PREDICTION_DIR + MODEL_NAME + "_test.npy", 'wb') as f:
    np.save(f, predictions)

  pred_df = pd.DataFrame(predictions, columns=["NEG", "NEU", "POS"])
  pred_df.to_csv(PREDICTION_DIR + MODEL_NAME + "_test.csv", index=None)


if __name__ == "__main__":
  main()

