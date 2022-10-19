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
    # magenta 6116,0,6116
    # dark orange 1000,550,0
    # light orange 3500,799,25
    # yellow 6116,1000,0
    # green 0,6116,0
    # green grass 500,2550,0
    # green dark 60,500,0

    # null, out of range
    if value < -(500e-3):
        return 6116,0,6116
    # green dark, enhance regrowth, high
    elif (value >= -(500e-3)) and (value < -(251e-3)):
        return 3500,799,25
    # green grass, enhance regrowth, low
    elif (value >= -(251e-3)) and (value < -(100e-3)):
        return 3500,799,25
    # green, unburned
    elif (value >= -(100e-3)) and (value < 99e-3):
        return 3500,799,25
    # yellow, low severity
    elif (value >= 99e-3) and (value < 269e-3):
        return 6116,1000,0
    # light orange, moderate-low severity
    elif (value >= 269e-3) and (value < 439e-3):
        return 6116,1000,0
    # dark orange, moderate-high severity
    elif (value >= 439e-3):
        return 0,6116,0
    else:
        print(value)
        return 6116,0,6116


def nbr_nbr_plus_calculation(path_img_source, path_mask_source, path_img_all_out_tif,path_img_color_mask_out_tif):


    img = rasterio.open(path_img_source)
    mask = rasterio.open(path_mask_source).read(1)

    #To find out number of bands in an image
    num_bands = img.count
    # print(img.meta)
    # print("Number of bands in the image = ", num_bands)

    # print(img.shape)

    meta = img.meta.copy()

    meta['dtype'] = 'int16'
    meta['count'] = 3

    # merge image, mask, and level of burn
    file_name_all = glob.glob(path_img_all_out_tif)
    if len(file_name_all)>0:
        print("ya existe el indice de este mapa")
    else:
        
        print("creando indice formato tif")
        r_band, b_band, g_band, nir_band, swir_band = return_nir_swir_band(img)

        print(b_band[0][0])

        nbr = (nir_band - swir_band) / (nir_band + swir_band)

        nbr_plus = (nir_band - swir_band - g_band - b_band) / (nir_band + swir_band + g_band + b_band)

        x, y = nbr.shape

        coloured_merge_red=nbr.copy()
        coloured_merge_green=nbr.copy()
        coloured_merge_blue=nbr.copy()

        for i in range(x):
            for j in range(y):
                r, g, b = return_index_nbr(nbr[i][j])
                if mask[i][j] == 1:
                    coloured_merge_red[i][j]=r
                    coloured_merge_green[i][j]=g
                    coloured_merge_blue[i][j]=b
                else:
                    coloured_merge_red[i][j]=r_band[i][j]
                    coloured_merge_green[i][j]=g_band[i][j]
                    coloured_merge_blue[i][j]=b_band[i][j]


        img_out=rasterio.open(path_img_all_out_tif, 'w', **meta)

        img_out.write(coloured_merge_red, 1)
        img_out.write(coloured_merge_green, 2)
        img_out.write(coloured_merge_blue, 3)
        img_out=None
    
    # color merge with mask
    file_name_color = glob.glob(path_img_color_mask_out_tif)
    if len(file_name_color)>0:
        print("ya existe el indice de este mapa")
    else:
        
        print("creando indice formato tif")
        _, b_band, g_band, nir_band, swir_band = return_nir_swir_band(img)

        print(b_band[0][0])

        nbr = (nir_band - swir_band) / (nir_band + swir_band)

        nbr_plus = (nir_band - swir_band - g_band - b_band) / (nir_band + swir_band + g_band + b_band)

        x, y = nbr.shape

        coloured_colormask_red=nbr.copy()
        coloured_colormask_green=nbr.copy()
        coloured_colormask_blue=nbr.copy()

        for i in range(x):
            for j in range(y):
                r, g, b = return_index_nbr(nbr[i][j])
                if mask[i][j] == 1:
                    coloured_colormask_red[i][j]=r
                    coloured_colormask_green[i][j]=g
                    coloured_colormask_blue[i][j]=b
                else:
                    coloured_colormask_red[i][j]=0
                    coloured_colormask_green[i][j]=0
                    coloured_colormask_blue[i][j]=0

        img_out=rasterio.open(path_img_color_mask_out_tif, 'w', **meta)

        img_out.write(coloured_colormask_red, 1)
        img_out.write(coloured_colormask_green, 2)
        img_out.write(coloured_colormask_blue, 3)

        img=None
        img_out=None
        nbr=None
        nbr_plus=None
        coloured_merge_red=None
        coloured_merge_green=None
        coloured_merge_blue=None
        coloured_colormask_red=None
        coloured_colormask_green=None
        coloured_colormask_blue=None



def process_all():
    data_path = './dataset'
    data_path_mask = './masks'
    data_path_all_out = './dataset_all_merge'
    data_path_mask_out = './dataset_color_mask_merge'
    input_filename = np.array(sorted(glob.glob(data_path + "/*.tif")))
    print(len(input_filename))
    
    start = timeit.default_timer()
    for file_name in input_filename:
        name=file_name.split('/')[-1]
        names=name.split('rgbnir0')
        tif_prefix=names[1].split('.t')
        mask_source_filename=data_path_mask+'/'+names[0]+'_0'+tif_prefix[0]+'_mask.t'+tif_prefix[1]
        output_file_name_all=data_path_all_out+'/'+names[0]+'all0'+names[1]
        output_file_name_color_mask=data_path_mask_out+'/'+names[0]+'colormask0'+names[1]
        nbr_nbr_plus_calculation(file_name,mask_source_filename,output_file_name_all,output_file_name_color_mask)

    end = timeit.default_timer()
    print("elapsed time: {}".format(end-start))




if __name__ == "__main__":
    process_all()