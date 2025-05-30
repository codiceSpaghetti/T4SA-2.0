import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

from vit_keras import vit
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import Callback
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL_DIR = "/workdir/data/t4sa2_0/version_6_2/models/"
DATASET_DIR = "/workdir/data/crawling_backup/"
DF_DIR = "/workdir/data/t4sa2_0/version_4/"

#MODEL_DIR = "/workdir/data/trained_models/"
#DATASET_DIR = "/workdir/data/images/"
#DF_DIR = "/workdir/data/dataset_T4SA/"

class WeightsSaver(Callback):
  def __init__(self, N):
    self.N = N
    self.batch = 0
    self.epoch = 0

  def on_batch_end(self, batch, logs={}):
    if self.batch % self.N == 0:
      name = MODEL_DIR + 'model_at_batch%03d.h5' % self.batch
      self.model.save(name)
      print(f"Saving model after {self.batch} batches")
    self.batch += 1

  def on_epoch_end(self, epoch, logs={}):
    name = MODEL_DIR + 'model_at_epoch%02d.h5' % self.epoch
    self.model.save(name)
    print(f"Saving model after {self.epoch} epoch")
    self.epoch += 1


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


def main():
  parser = argparse.ArgumentParser(description='Fine tune the boosted model on the FI dataset.')
  parser.add_argument("--batch_size", "-batch", type=int, default=32, help="Batch size of training")
  parser.add_argument("--epochs", type=int, default=5, help="Number of epochs of training")
  parser.add_argument("--image_size", type=int, default=384, help="Size of images in input to the model")

  args = parser.parse_args()

  BATCH_SIZE = args.batch_size
  EPOCHS = args.epochs
  IMAGE_SIZE = args.image_size

  print("Reading csv files with training and validation set...")
  complete_annot = pd.read_csv(DF_DIR + "merge_complete.csv")
  train_annot = pd.read_csv(DF_DIR + "merge_train.csv")
  val_annot = pd.read_csv(DF_DIR + "merge_val.csv")

  train_dataset = build_data_pipeline(train_annot, IMAGE_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
  valid_dataset = build_data_pipeline(val_annot, IMAGE_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

  y = complete_annot["class"].apply(int).tolist()

  classes = np.unique(y)
  class_weight = compute_class_weight("balanced", classes=classes, y=y)
  class_weight_dict = dict(zip(classes, class_weight))

  print("Using class weights:", class_weight_dict)

  vit_model = vit.vit_b16(
    image_size=IMAGE_SIZE,
    pretrained=True,
    include_top=False,
    pretrained_top=False)

  vit_model.trainable = False

  model = tf.keras.Sequential([
    vit_model,
    tf.keras.layers.Dense(32, activation=tfa.activations.gelu),
    tf.keras.layers.Dense(3, activation='softmax')
  ], name='vision_transformer')

  model.summary()

  learning_rate = 1e-4
  optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)

  model.compile(optimizer=optimizer,
                loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
                metrics=['accuracy'])

  print("Training the top classifier while freezing the vit model...")
  model.fit(x=train_dataset,
            epochs=1,
            callbacks=[WeightsSaver(1000)],
            class_weight=class_weight_dict,
            verbose=1)

  model.save(MODEL_DIR + 'classifier_trained.h5')

  print("Loading the trained classifier")
  model = tf.keras.models.load_model(MODEL_DIR + "classifier_trained.h5")

  model.trainable = True
  model.summary()

  print("Training the model fully unfreezed...")

  model.fit(x=train_dataset,
            validation_data=valid_dataset,
            epochs=EPOCHS,
            callbacks=[WeightsSaver(100)],
            class_weight=class_weight_dict,
            verbose=1)


if __name__ == "__main__":
  main()
