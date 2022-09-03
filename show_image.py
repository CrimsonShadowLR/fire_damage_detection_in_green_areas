import argparse
import glob
import os
import pickle
from image_utils import masks_to_colorimg, pred_to_colorimg, reverse_transform

import numpy as np
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

from models import UNet11
from model_utils import load_model, run_model
from preprocessing import preprocess_image

def show_sample_images():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    # model-related variables
    arg('--model-path', type=str, help='Model path')
    arg('--num-picture', type=int, default=3, help='Number of sample pictures')
    arg('--dataset', type=str, help='burn: burn segmentation')

    # image-related variables
    arg('--image-patches-dir', type=str, default='./data/dataset/fs7mtkg2wk-4/images', help='satellite image patches directory')
    arg('--masks-dir', type=str, default='./data/dataset/fs7mtkg2wk-4/masks', help='numPy masks directory')
    arg('--npy-dir', type=str, default='./data/dataset/fs7mtkg2wk-4/split_npy', help='numPy preprocessed patches directory')

    args = parser.parse_args()

    modelname = args.model_path[args.model_path.rfind("/") + 1:args.model_path.rfind(".pth")]

    model = load_model(args.model_path, UNet11)

    print("Testing {} on {} samples".format(modelname, args.num_picture))
    print(args.num_picture)

    # Select sample pictures
    images_filenames = np.array(sorted(glob.glob(args.npy_dir + "/*.npy")))
    print(len(images_filenames))
    sample_filenames = np.random.choice(images_filenames, args.num_picture)

    fig = plt.figure(figsize=(10, 7.5 * args.num_picture))

    for idx, filename in enumerate(sample_filenames):
        print("Loading sample input {}".format(idx))
        image = pickle.load(open(filename, "rb"))
        image = preprocess_image(image)

        print("Running model for sample {}".format(idx))
        pred = run_model(image, model)

        basename=filename.rfind("/")
        print(basename)
        fl_split=basename.split('.')
        mask_path = args.masks_dir+ "/." + fl_split[1]+ "_mask."+ fl_split[2]
        y = pickle.load(open(mask_path, "rb"))
        print("Get mask for sample {}".format(idx))

        fig.add_subplot(args.num_picture, 3, idx * 3 + 1)
        plt.imshow(reverse_transform(image.cpu().numpy()[0]))
        print("Add plot for sample input {}".format(idx))

        fig.add_subplot(args.num_picture, 3, idx * 3 + 2)
        plt.imshow(masks_to_colorimg(y))
        print("Add plot for sample mask {}".format(idx))

        fig.add_subplot(args.num_picture, 3, idx * 3 + 3)
        plt.imshow(pred_to_colorimg(pred.cpu().numpy()))
        print("Add plot for sample pred {}".format(idx))

    if not os.path.exists("test"):
        os.mkdir("test")

    if not os.path.exists("test/{}".format(args.dataset)):
        os.mkdir("test/{}".format(args.dataset))

    plt.savefig("test/{}/test_{}_samples_{}.png"
                .format(args.dataset,
                        args.num_picture,
                        modelname))


if __name__ == "__main__":
    show_sample_images()
