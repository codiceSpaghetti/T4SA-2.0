# -*- coding: utf-8 -*-
import requests
import tensorflow as tf
import tensorflow_addons as tfa

from vit_keras import vit
from tensorflow import keras
from tensorflow.keras import layers


BENCHMARK_DIR = "dataset/benchmark/"

TWITTER_1_DIR = BENCHMARK_DIR + "Twitter Testing Dataset I/"
TWITTER_2_DIR = BENCHMARK_DIR + "Twitter Testing Dataset II/"
EMOTION_ROI_DIR = BENCHMARK_DIR + "EmotionROI/"
FI_DIR = BENCHMARK_DIR + "FI/"

image_size_mapper = {"b32": 224, "b16": 384, "l16": 384, "l32": 384, "b32_384": 384}

map_dataset_to_folder = {TWITTER_1_DIR + "3agree.csv": TWITTER_1_DIR + "images/",
                         TWITTER_1_DIR + "4agree.csv": TWITTER_1_DIR + "images/",
                         TWITTER_1_DIR + "5agree.csv": TWITTER_1_DIR + "images/",
                         TWITTER_2_DIR + "twitter_testing_2.csv": TWITTER_2_DIR + "images/",
                         FI_DIR + "FI.csv": FI_DIR + "images/",
                         EMOTION_ROI_DIR + "emotion_ROI_test.csv": EMOTION_ROI_DIR + "images/",
                         EMOTION_ROI_DIR + "emotion_ROI_train.csv": EMOTION_ROI_DIR + "images/",
                         EMOTION_ROI_DIR + "emotion_ROI_complete.csv": EMOTION_ROI_DIR + "images/"}

map_model_to_url = {"unb_T4SA2.0":        "https://drive.google.com/uc?export=download&id=1t_PYWXRkxhPJxY0stGLNiNsYiqvxAioM",
                    "merged_T4SA":        "https://drive.google.com/uc?export=download&id=1jemKao4kxywpuysdhIYu81Txj5ICO2cn",
                    "bal_T4SA2.0":        "https://drive.google.com/uc?export=download&id=12KLZ31M-KG_hqXHmkVKG11ZZCZyFr--4",
                    "bal_flat_T4SA2.0":   "https://drive.google.com/uc?export=download&id=1iO9FugxVSbjao6pyvFbJHUzjzpIf2XY0",
                    "ViT_L32":            "https://drive.google.com/uc?export=download&id=1vLyoLiQDYuKyX8JFImYImLsb-otqJFcT",
                    "ViT_L16":            "https://drive.google.com/uc?export=download&id=1LpCa1Pm_i921hsTZPVsc9L7sovqqOXNG",
                    "ViT_B32":            "https://drive.google.com/uc?export=download&id=1rdCi0X42iU6mjddggFJmlhJdqki50jPE",
                    "ViT_B16":            "https://drive.google.com/uc?export=download&id=1OrlpIY-NazGlwCeHAGfAATTOfZezyh_7",
                    "B-T4SA_1.0":         "https://drive.google.com/uc?export=download&id=1SteBWCe60TystYr-VdBrqTtyMlLd9FK4",
                    "B-T4SA_1.0_upd":     "https://drive.google.com/uc?export=download&id=1eyKSbT3PXy-cNtAUYrQU2nrn1H57pSIY",
                    "B-T4SA_1.0_upd_filt":"https://drive.google.com/uc?export=download&id=1iI0GTs0wXGtjybMVoSCNRImw0o_HiPeZ",
                    "boosted_model":      "https://drive.google.com/uc?export=download&id=1Nibw6Us5gqek3dYFYoJ8BXGRA0FUb7VR",
                    }

map_benchmark_to_folder = {"3agree.csv": TWITTER_1_DIR,
                           "4agree.csv": TWITTER_1_DIR,
                           "5agree.csv": TWITTER_1_DIR,
                           "twitter_testing_2.csv": TWITTER_2_DIR,
                           "FI_complete.csv": FI_DIR,
                           "FI_test.csv": FI_DIR,
                           "emotion_ROI_test.csv": EMOTION_ROI_DIR,
                           "emotion_ROI_train.csv": EMOTION_ROI_DIR,
                           "emotion_ROI_complete.csv": EMOTION_ROI_DIR}


def change_labels(label):
  if label == 2:
    label = 1

  return label


def download_file(url, local_filename):
  with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(local_filename, 'wb') as f:
      for chunk in r.iter_content(chunk_size=8096):
        if chunk:
          f.write(chunk)

  return local_filename



def get_dataset(annot, vit_model, dataset_name, batch_size=32):
  '''Returns a tf.data.Dataset that maps image and labels taken from the dataframe in input.'''
  x = annot['path'].to_list()
  labels = annot['class'].apply(int).to_list()
  y = [elem if elem == 0 else 2 for elem in labels]

  # Buld a tensorflow dataset
  data = tf.data.Dataset.from_tensor_slices((x, y))

  # Map the dataset with the preprocessing function
  data = data.map(
    lambda x, y: (load_image_tf(x, vit_model, dataset_name), y),  # path -> image, keep y unaltered
    num_parallel_calls=tf.data.AUTOTUNE,  # load in parallel
    deterministic=True  # keep the order
  ).batch(batch_size)

  return data


def load_image_tf(path, vit_model, dataset_name):
  '''Decodes the image specified by the path in input and applies some preprocessing to it.'''
  image_data = tf.io.read_file(map_benchmark_to_folder[dataset_name] + "images/" + path)  # read image file
  image = tf.image.decode_image(image_data, channels=3, expand_animations=False)  # decode image data as RGB (do not load whole animations, i.e., GIFs)
  image = tf.image.resize(image, (image_size_mapper[vit_model], image_size_mapper[vit_model]))  # resize

  if vit_model == "l16" or vit_model == "l32" or vit_model == "b16":
    image = vit.preprocess_inputs(image)

  return image


def get_5_fold_architecture(model_name):
  '''Returns a tf.data.Dataset that maps image and labels taken from the dataframe in input.'''
  # Load the base ViT model
  vit_model = vit.vit_l16(
      image_size=384,
      pretrained=True,
      include_top=False,
      pretrained_top=False
  )

  # Add the classification head
  model = tf.keras.Sequential([
    vit_model,
    layers.Dense(32, activation=tfa.activations.gelu),
    layers.Dense(3, activation='softmax')
  ],
    name='vision_transformer')

  mask = tf.convert_to_tensor([1.0, 0.0000001, 1.0], dtype=tf.float32)
  mask = tf.expand_dims(tf.cast(mask, dtype=tf.float32), axis=0)

  inputs = model.inputs
  x = model(inputs)
  outputs = tf.keras.layers.Multiply()([x, mask])

  model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

  model.trainable = False

  model.load_weights(model_name)  # load the saved weights

  return model


def get_dataset(annot, dataset_name, vit_model="l16", batch_size=32):
  '''Returns a tf.data.Dataset that maps image and labels taken from the dataframe in input.'''
  x = annot['path'].to_list()
  labels = annot['class'].apply(int).to_list()
  y = [elem if elem == 0 else 2 for elem in labels]

  # Buld a tensorflow dataset
  data = tf.data.Dataset.from_tensor_slices((x, y))

  # Map the dataset with the preprocessing function
  data = data.map(
    lambda x, y: (load_image_tf(x, vit_model, dataset_name), y),  # path -> image, keep y unaltered
    num_parallel_calls=tf.data.AUTOTUNE,  # load in parallel
    deterministic=True  # keep the order
  ).batch(batch_size)

  return data


def get_vit_type(model):
  if model == "boosted_model" or model == "ViT_L16":
    vit_type = "l16"
  elif model == "ViT_L32":
    vit_type = "l32"
  elif model == "ViT_B16":
    vit_type = "b16"
  else:
    vit_type = "b32"

  return vit_type


def get_model_architecture(vit_model):
  '''Returns a tf.data.Dataset that maps image and labels taken from the dataframe in input.'''
  # Load the base ViT model
  if vit_model == "l16":
    vit_model = vit.vit_l16(
      image_size=image_size_mapper[vit_model],
      pretrained=True,
      include_top=False,
      pretrained_top=False
    )
  elif vit_model == "l32":
    vit_model = vit.vit_l32(
      image_size=image_size_mapper[vit_model],
      pretrained=True,
      include_top=False,
      pretrained_top=False
    )
  elif vit_model == "b16":
    vit_model = vit.vit_b16(
      image_size=image_size_mapper[vit_model],
      pretrained=True,
      include_top=False,
      pretrained_top=False
    )
  elif vit_model == "b32":
    vit_model = vit.vit_b32(
      image_size=image_size_mapper[vit_model],
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

  return model

