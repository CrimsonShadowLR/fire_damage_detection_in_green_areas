import numpy as np
import torch
import os
import pickle
import rasterio

from torch.utils.data.dataset import Dataset


class LandSat8SegmentationDataset(Dataset):
    def __init__(self, filenames, mask_dir, dataset, transform=None, mode='train', limit=None):
        self.filenames = filenames
        self.mask_dir = mask_dir
        self.dataset = dataset
        self.transform = transform
        self.mode = mode
        self.limit = limit

    def __len__(self):
        return len(self.filenames) if self.limit is None else self.limit

    def __getitem__(self, idx):

        if self.limit is None:
            filename = self.filenames[idx]
        else:
            filename = np.random.choice(self.filenames)

        basename = filename[filename.rfind("/") + 1:]
        mask_filename = os.path.join(self.mask_dir, basename)
        img = load_image(filename)

        if self.mode == 'train':
            mask = load_mask(mask_filename)

            img, mask = self.transform(img, mask)

            return to_float_tensor(img), torch.from_numpy(np.expand_dims(mask, 0)).float()

        else:
            mask = np.zeros(img.shape[:2])
            img, mask = self.transform(img, mask)

            return to_float_tensor(img), str(filename)


def to_float_tensor(img):
    img = torch.from_numpy(np.moveaxis(img, -1, 0)).float()
    return img


def load_image(path):  # Input:CH,H,W  Output:H,W,CH
    img = rasterio.open(path)
    img = img.read()
    img = img.transpose((1, 2, 0))
    return img


def load_mask(path):  # Input:H,W  Output:H,W,CH

    x = path.split('.')
    mask = pickle.load(open('.'+x[1]+"_mask."+x[2], "rb"))
    mask = np.squeeze(mask, axis=(0,))
    mask = np.float32(mask)
    return mask
