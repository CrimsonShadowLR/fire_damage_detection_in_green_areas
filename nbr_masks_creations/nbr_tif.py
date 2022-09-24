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

def nbr_nbr_plus_calculation(path_img_source,path_img_out_tif):

    img = rasterio.open(path_img_source)

    #To find out number of bands in an image
    num_bands = img.count
    # print(img.meta)
    # print("Number of bands in the image = ", num_bands)

    # print(img.shape)

    meta = img.meta.copy()

    meta['dtype'] = 'float32'
    meta['count'] = 2

    print(meta)

    file_name = glob.glob(path_img_out_tif)
    if len(file_name)>0:
        print("ya existe el indice de este mapa")
    else:
        
        print("creando indice formato tif")
        _, b_band, g_band, nir_band, swir_band = return_nir_swir_band(img)

        nbr = (nir_band - swir_band) / (nir_band + swir_band)

        nbr_plus = (nir_band - swir_band - g_band - b_band) / (nir_band + swir_band + g_band + b_band)

        img_out=rasterio.open(path_img_out_tif, 'w', **meta)

        img_out.write(nbr, 1)
        img_out.write(nbr_plus, 2)

        img=None
        img_out=None
        nbr=None
        nbr_plus=None

def calculate_all():
    data_path = './dataset'
    data_path_out = './dataset_nbr'
    input_filename = np.array(sorted(glob.glob(data_path + "/*.tif")))
    print(len(input_filename))
    
    start = timeit.default_timer()
    for file_name in input_filename:
        name=file_name.split('/')[-1]
        names=name.split('rgbnir0')
        output_file_name=data_path_out+'/'+names[0]+'nbr0'+names[1]
        nbr_nbr_plus_calculation(file_name,output_file_name)

    end = timeit.default_timer()
    print("elapsed time: {}".format(end-start))




if __name__ == "__main__":
    calculate_all()