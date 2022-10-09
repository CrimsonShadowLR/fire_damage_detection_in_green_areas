# Librerias a utilizar
import os
import glob

from pathlib import Path
import numpy as np
import rasterio
import timeit

def return_nir_swir_band(img):
    b_band = img.read(1).astype('f4')
    g_band = img.read(2).astype('f4')
    r_band = img.read(3).astype('f4')
    nir_band = img.read(4).astype('f4')
    swir_band = img.read(5).astype('f4')
    return (r_band, b_band, g_band, nir_band, swir_band)

def return_index_nbr(value):
    # null, out of range
    if value < -(500e-3):
        return 0,0,0
    # green dark, enhance regrowth, high
    elif (value >= -(500e-3)) and (value < -(251e-3)):
        return 118,136,52
    # green grass, enhance regrowth, low
    elif (value >= -(251e-3)) and (value < -(100e-3)):
        return 168,192,80
    # green, unburned
    elif (value >= -(100e-3)) and (value < 99e-3):
        return 11,227,68
    # yellow, low severity
    elif (value >= 99e-3) and (value < 269e-3):
        return 248,252,17
    # light orange, moderate-low severity
    elif (value >= 269e-3) and (value < 439e-3):
        return 248,176,64
    # dark orange, moderate-high severity
    elif (value >= 439e-3) and (value < 659e-3):
        return 247,104,26
    # magenta, high severity
    else:
        return 168,0,209

def nbr_nbr_plus_calculation(path_img_source,path_img_out_tif):


    img = rasterio.open(path_img_source)

    #To find out number of bands in an image
    num_bands = img.count
    # print(img.meta)
    # print("Number of bands in the image = ", num_bands)

    # print(img.shape)

    meta = img.meta.copy()

    meta['dtype'] = 'int16'
    meta['count'] = 3

    print(meta)

    file_name = glob.glob(path_img_out_tif)
    if len(file_name)>0:
        print("ya existe el indice de este mapa")
    else:
        
        print("creando indice formato tif")
        _, b_band, g_band, nir_band, swir_band = return_nir_swir_band(img)

        print(b_band[0][0])

        nbr = (nir_band - swir_band) / (nir_band + swir_band)

        nbr_plus = (nir_band - swir_band - g_band - b_band) / (nir_band + swir_band + g_band + b_band)

        x, y = nbr.shape

        coloured_nbr_red=nbr.copy()
        coloured_nbr_green=nbr.copy()
        coloured_nbr_blue=nbr.copy()

        for i in range(x):
            for j in range(y):
                r, g, b = return_index_nbr(nbr[i][j])
                coloured_nbr_red[i][j]=r
                coloured_nbr_green[i][j]=g
                coloured_nbr_blue[i][j]=b

        img_out=rasterio.open(path_img_out_tif, 'w', **meta)

        img_out.write(coloured_nbr_red, 1)
        img_out.write(coloured_nbr_green, 2)
        img_out.write(coloured_nbr_blue, 3)

        img=None
        img_out=None
        nbr=None
        nbr_plus=None
        coloured_nbr_red=None
        coloured_nbr_green=None
        coloured_nbr_blue=None

def calculate_all():
    data_path = './dataset'
    data_path_out = './dataset_color_nbr'
    input_filename = np.array(sorted(glob.glob(data_path + "/*.tif")))
    print(len(input_filename))
    
    start = timeit.default_timer()
    for file_name in input_filename:
        name=file_name.split('/')[-1]
        names=name.split('rgbnir0')
        output_file_name=data_path_out+'/'+names[0]+'colornbr0'+names[1]
        nbr_nbr_plus_calculation(file_name,output_file_name)
        

    end = timeit.default_timer()
    print("elapsed time: {}".format(end-start))




if __name__ == "__main__":
    calculate_all()