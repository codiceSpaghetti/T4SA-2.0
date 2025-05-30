import os
import faiss
import argparse
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from sklearn.preprocessing import normalize

BASE_DIR = "/datadir/t4sa2_0/"
DATASET_DIR = "/datadir/crawling/"
FILTERING_DIR = "/workdir/data/filtering/"
SAVAGE_DIR = FILTERING_DIR + "duplicates/"


def save_backup(distances, indexes, file_list, threshold,is_finished=False):
  # Normalization of the distances to fix them in the range [0,1]
  distances = normalize(distances)

  if is_finished:
    with open(SAVAGE_DIR + '/_distance_end.npy', 'wb') as f:
      np.save(f, distances)

    with open(SAVAGE_DIR + '/_indexes_end.npy', 'wb') as f:
      np.save(f, indexes)
  else:
    with open(SAVAGE_DIR + '/_distance_backup.npy', 'wb') as f:
      np.save(f, distances)

    with open(SAVAGE_DIR + '/_indexes_backup.npy', 'wb') as f:
      np.save(f, indexes)

  img_with_duplicate = []
  list_of_duplicates = []

  # Iterate for all the images
  checked = set()

  for i in range(len(distances)):
    index_vector = indexes[i]  # Get the index vector of the current image
    if index_vector[0] in checked:
      continue

    distance_vector = distances[i]  # Get the distance vector of the current image
    mask_duplicates = distance_vector < threshold

    index_vector = index_vector[mask_duplicates]
    index_vector = [ele for ele in index_vector if ele not in checked]

    if len(index_vector) > 1:
      img_with_duplicate.append(file_list[index_vector[0]])
      checked.update(index_vector)

      duplicates = [file_list[int(i)] for i in index_vector[1:]]
      list_of_duplicates.append(duplicates)

  count_dup = 0
  for elem in list_of_duplicates:
    count_dup += len(elem)

  print("Number of duplicates found:", count_dup)
  duplicate_df = pd.DataFrame({"duplicated image": img_with_duplicate, "duplicates": list_of_duplicates})

  if is_finished:
    duplicate_df.to_csv(SAVAGE_DIR + "dup_" + str(SLICE[0]) + "_" + str(SLICE[1]) + "_end.csv", index=None)
  else:
    duplicate_df.to_csv(SAVAGE_DIR + "dup_" + str(SLICE[0]) + "_" + str(SLICE[1]) + "_backup.csv", index=None)

def main():
  parser = argparse.ArgumentParser(description='Program to find duplicates using image features.')
  parser.add_argument("--num_nn",  type=int, default=100, help="Number of Nearest Neighbor to return in the search")
  parser.add_argument("--n_probe", type=int, default=10, help="Number of contiguous cluster in which extend the search")
  parser.add_argument("--n_list", type=int, default=1000, help="Number of posting lists = number of clusters in k-means")
  parser.add_argument("--threshold", type=float, default=0.15, help="Maximum distance between features to be duplicates")

  args = parser.parse_args()

  NUM_NN = args.num_nn
  N_PROBE = args.n_probe
  N_LIST = args.n_list # number of posting lists = number of clusters in k-means
  THRESHOLD = args.threshold



  all_imgs = pd.read_csv(FILTERING_DIR + "all_imgs_filtered.csv")

  with open(FILTERING_DIR + "dataset_features.npy", 'rb') as f:
    features = np.load(f)

  file_list = all_imgs['path'].to_list()

  d = features.shape[1]

  quantizer = faiss.IndexFlatL2(d)  # this is the index that will contain the
                                    # clusters and will be used to search the
                                    # nearest centroids

  index = faiss.IndexIVFFlat(quantizer, d, N_LIST)

  print("Training the index...")
  index.train(features)  # this index must be trained (run k-means) to be used
  print("Training end, is_trained: ", index.is_trained)

  index.add(features)  # now we can add the data

  # return the k-NN
  k = NUM_NN
  index.nprobe = N_PROBE
  print("Starting the sequential search to find the 30-NN of each feature...")

  q = features[0]
  q = np.expand_dims(q, axis=0)
  distances, indexes = index.search(q, k)  # D: a (N,k)-shaped matrix where D[i,j] contains the L2 distance of the j-th neighbor of the i-th query.
                                            # I: a (N,k)-shaped matrix where I[i,j] contains the index of the j-th neighbor of the i-th query.

  for i, feature in tqdm(enumerate(features[1:]), total=features.shape[0]):
    feature = np.expand_dims(feature, axis=0)
    dist, ind = index.search(feature, k)
    distances = np.append(distances, dist, axis=0)
    indexes = np.append(indexes, ind, axis=0)

    if i != 0 and i%10000 == 0:
      print(f"Saving backup after {i} images...")
      save_backup(distances, indexes, file_list, THRESHOLD)

  print(f"End of the search saving the final file...")
  save_backup(distances, indexes, file_list, THRESHOLD, is_finished=True)


if __name__ == "__main__":
  main()