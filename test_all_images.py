import argparse
import glob
import os
import pickle

import numpy as np
import matplotlib as mpl

from data_loader import load_image
from image_utils import masks_to_colorimg, pred_to_colorimg, reverse_transform
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

from models import UNet11
from preprocessing import preprocess_image
from model_utils import load_model, run_model


def test_all_images():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    
    # model-related variables model_path='./trained_models/model_25_percent_UNet11_Unet11_350epochs',
    arg('--model-path', type=str, default='./trained_models/model_25_percent_UNet11_Unet11_350epochs', help='Model path')
    arg('--dataset', type=str, default='burn', help='burn: burn segmentation')

    # image-related variables
    arg('--masks-dir', type=str, default='./data/dataset/fs7mtkg2wk-4/masks', help='numPy masks directory')
    arg('--images-dir', type=str, default='./data/dataset/fs7mtkg2wk-4/images', help='numPy preprocessed patches directory')

    args = parser.parse_args()

    model = load_model(args.model_path, UNet11)

    if not os.path.exists("./test_all"):
        os.mkdir("./test_all")

    # date str
    now = datetime.now()
    time_str=now.strftime("%H:%M:%S")

    # Select sample pictures
    images_filenames = np.array(sorted(glob.glob(args.images_dir + "/*.tif")))

    for filename in tqdm(images_filenames):
        
        print(filename, 1)

        fig = plt.figure(figsize=(10, 10))

        image = load_image(filename)
        image = preprocess_image(image)

        pred = run_model(image, model)

        mask_path = os.path.join(args.masks_dir, filename[filename.rfind("/") + 1:])
        y = pickle.load(open(mask_path, "rb"))

        fig.add_subplot(1, 3, 1)
        plt.imshow(reverse_transform(image.cpu().numpy()[0]))

        fig.add_subplot(1, 3, 2)
        plt.imshow(masks_to_colorimg(y))

        fig.add_subplot(1, 3, 3)
        plt.imshow(pred_to_colorimg(pred.cpu().numpy()))
        
        print(filename, 5)
        plt.savefig(os.path.join("./test_all",
                                 filename[filename.rfind("/") + 1:filename.rfind(".")] + ".png"))

        plt.clf()
        plt.close(fig)


if __name__ == "__main__":
    test_all_images()
