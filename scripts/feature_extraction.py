import os
import faiss
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm.auto import tqdm
from sklearn.preprocessing import normalize

DATASET_DIR = "/workdir/data/crawling_backup/"
FILTERING_DIR = "/workdir/data/filtering/"


def load_image_tf(path, image_size):
  image_data = tf.io.read_file(DATASET_DIR + path)  # read image file
  image = tf.image.decode_image(image_data, channels=3, expand_animations=False)  # decode image data as RGB (do not load whole animations, i.e., GIFs)
  image = tf.image.resize(image, (image_size, image_size))  # resize
  image = tf.keras.applications.resnet50.preprocess_input(image)

  return image


def main():
  parser = argparse.ArgumentParser(description='Program to extract features from images.')
  parser.add_argument("--batch_size", "-batch", type=int, default=256, help="Batch size of testing")
  parser.add_argument("--image_size", type=int, default=224, help="Size of images in input to the model")

  args = parser.parse_args()

  BATCH_SIZE = args.batch_size
  IMAGE_SIZE = args.image_size

  all_imgs = pd.read_csv(FILTERING_DIR + "all_imgs_filtered.csv")

  file_list = all_imgs['path'].to_list()
  n_images = len(file_list)

  print("Extracting features for", n_images, "files...")

  image_batches_ds = tf.data.Dataset.from_tensor_slices((file_list))
  image_batches_ds = image_batches_ds.map(
    lambda x: (load_image_tf(x, IMAGE_SIZE)),  # path -> image
    num_parallel_calls=tf.data.AUTOTUNE,  # load in parallel
    deterministic=True  # keep the order (we will shuffle afterward if needed)
  ).batch(BATCH_SIZE)  # batch the dataset

  INPUT_SIZE = (IMAGE_SIZE, IMAGE_SIZE, 3)

  base_model = tf.keras.applications.resnet50.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=INPUT_SIZE,
    pooling="max"
  )

  inputs = tf.keras.Input(shape=INPUT_SIZE)
  res_net = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('conv4_block6_out').output,
                           name="res_net")
  x = res_net(inputs)
  output = tf.keras.layers.GlobalMaxPooling2D()(x)
  feature_extractor = tf.keras.Model(inputs=inputs, outputs=output)

  feature_extractor.summary()

  temp = []
  n_batches = n_images // BATCH_SIZE  # integer division

  for i, batch in tqdm(enumerate(image_batches_ds.as_numpy_iterator()), total=n_batches):
    image_feature_vectors = feature_extractor(batch).numpy()  # apply the model to each batch

    for image_feature_vector in image_feature_vectors:  # iterator over batch of features
      temp.append(image_feature_vector)

    if i != 0 and (i * BATCH_SIZE) % 100000 == 0:
      print(f"Saving backup after {i * BATCH_SIZE} elements...")
      features = np.asarray(temp)
      with open(FILTERING_DIR + 'features_backup.npy', 'wb') as f:
        np.save(f, features)

  features = np.asarray(temp)

  with open(FILTERING_DIR + 'dataset_features.npy', 'wb') as f:
    np.save(f, features)

  all_imgs.to_csv(FILTERING_DIR + "dataset_features.csv", index=None)


if __name__ == "__main__":
  main()
