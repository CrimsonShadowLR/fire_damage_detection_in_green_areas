import pickle
from pathlib import Path
from data_loader import to_float_tensor

import numpy as np
import torch
from torch.autograd import Variable
from transform import DualCompose, ImageOnly, Normalize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
    Implementation from https://github.com/jesseniagonzalezv/App_segmentation_water_bodies/
    &                   https://github.com/csosapezo/socioeconomic-satellite-classification/
"""

def find_max(im_pths):
    min_pixel = []
    max_pixel = []
    size = len(im_pths)

    for i in im_pths:
        img = pickle.load(open(i, "rb"))

        img = img.transpose((1, 2, 0))

        min_pixel.append(np.min(img))
        max_pixel.append(np.max(img))

    return np.min(min_pixel), np.max(max_pixel), size

def cal_dir_stat(im_pths, max_value, channel_num):
    pixel_num = 0
    aux = None
    channel_sum = np.zeros(channel_num)
    channel_sum_squared = np.zeros(channel_num)

    for path in im_pths:
        im = pickle.load(open(path, "rb"))

        if channel_num != 4:
            aux = im[:][-1]

        im = im / max_value

        if channel_num != 4:
            im[-1] = aux

        im = im.transpose((1, 2, 0))
        pixel_num += (im.size / channel_num)
        channel_sum += np.sum(im, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

    rgb_mean = channel_sum / pixel_num
    rgb_std = np.sqrt(channel_sum_squared / pixel_num - np.square(rgb_mean))

    return rgb_mean, rgb_std

def meanstd(root, rootdata='data_VHR', channel_num='8'):  # name_file,
    data_path = Path(rootdata)

    minimo_pixel_all, maximo_pixel_all, size_all = find_max(root)
    mean_all, std_all = cal_dir_stat(root, maximo_pixel_all, channel_num)

    print('All:', str(data_path), size_all, 'min ', np.min(minimo_pixel_all), 'max ', maximo_pixel_all)  # 0-1

    print("mean:{}\nstd:{}".format(mean_all, std_all))
    return maximo_pixel_all, mean_all, std_all

def preprocess_image(img):
    """Normaliza y transforma la imagen en un tensor apto para ser procesado por la red neuronal de segmentación de
    cuerpos de agua.
    Dimensiones: entrada: (8,256,256); salida: (1,8,256,256)
    :param img: imagen por preprocesar
    :type img: np.ndarray
    """
    img = img.transpose((1, 2, 0))
    image_transform = transform_function()
    img_for_model = image_transform(img)[0]
    img_for_model = Variable(to_float_tensor(img_for_model), requires_grad=False)
    img_for_model = img_for_model.unsqueeze(0).to(device)

    return img_for_model


def transform_function():
    """Función de normalización para una imagen satelital."""
    image_transform = DualCompose([ImageOnly(
            Normalize(mean=[0.193204732, 0.171372226, 0.150260641, 0.131608721, 0.305681679, 0.188308873, 0.105871855, 0.0485840778],
                      std=[0.107186223, 0.112999831, 0.111625422, 0.118835341, 0.13037901, 0.101496176, 0.0789984236, 0.0855568126]))])
    
    return image_transform
