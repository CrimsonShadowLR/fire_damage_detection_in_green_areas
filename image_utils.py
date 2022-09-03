import numpy as np


def reverse_transform(inp):
    inp = inp.transpose(1, 2, 0)

    mean = np.array([0.09444648, 0.08571006, 0.10127277, 0.09419213])
    std = np.array([0.03668221, 0.0291096, 0.02894425, 0.03613606])

    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp / inp.max())
    inp = (inp * 255).astype(np.uint8)
    inp = inp.transpose(2, 0, 1)
    inp = inp[:3]
    inp = inp.transpose(1, 2, 0)
    return inp


def masks_to_colorimg(mask):
    image = np.zeros(shape=[3, 512, 512])

    image[0] = mask[:] * 255
    image[1] = mask[:] * 255
    image[2] = mask[:] * 255

    image = image.transpose((1, 2, 0))
    image = image.astype('uint8')

    return image

def pred_to_colorimg(mask):
    mask = mask[0]
    new_mask = np.zeros(shape=[512, 512])
    for x in range(512):
        for y in range(512):
            new_mask[y, x] = int(mask[0, y, x] > 0.5)

    image = masks_to_colorimg(new_mask)
    return image
