import numpy as np
import os
import pickle

import rasterio


def train_val_test_split(dataset_size, val_percent, test_percent):
    """
    Split a dataset in train and validation subset

    :param dataset_size: dataset size
    :type dataset_size: int

    :param val_percent: percentage of the dataset assigned to the validation subset
    :type val_percent: float

    :param test_percent: percentage of the dataset assigned to the test subset
    :type test_percent: float


    :return: train and validation subset indices
    :rtype: (list[int], list[int], list[int])
    """

    np.random.seed(0)

    indices = np.random.permutation(np.arange(dataset_size))
    val_size = int(val_percent * dataset_size)
    test_size = int(test_percent * dataset_size)

    train_start = val_size + test_size

    val_set_indices = indices[:val_size]
    test_set_indices = indices[val_size:train_start]
    train_set_indices = indices[train_start:]

    return train_set_indices, val_set_indices, test_set_indices

def save_npy(image_file_names, output_path, model, masks_dir):
    np_image_names = []

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for name in image_file_names:
        with rasterio.open(name) as dataset:
            raster = dataset.read()

            dot = name.rfind(".")
            slash = name.rfind("/") + 1
            
            np_name = str(os.path.join(output_path, name[slash:dot] + ".npy"))
            np_image_names.append(np_name)

            pickle.dump(raster, open(np_name, "wb"))

    return np_image_names