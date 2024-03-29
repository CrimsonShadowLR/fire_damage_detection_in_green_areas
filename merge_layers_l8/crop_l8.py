"""
Split images

Input:
.TIF Images of approximately 6000x6000 size 

Output:
.TIF Images of 512x512x4 (RGBNIR)
The complete images have some parts in blacks, which will be not considered if there is more than 15% of black parts in the images.

"""
from pathlib import Path
import rasterio as rio
from rasterio import windows
import numpy as np
import os
from itertools import product
import timeit
import csv

def splits_images(in_path,out_path,input_filename,output_filename,output_filename_npy,output_filename_npyblack):
    coordinates='{}-{},{}-{}'

    def get_tiles(ds, width=512, height=512):
        nols, nrows = ds.meta['width'], ds.meta['height']
        offsets = product(range(0, nols, width), range(0, nrows, height))
        big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
        for col_off, row_off in  offsets:
            window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
            transform = windows.transform(window, ds.transform)  #split
            yield window, transform


    with rio.open(input_filename) as inds:
        meta = inds.meta.copy()


        for window, transform in get_tiles(inds):

            meta['transform'] = transform
            meta['width'], meta['height'] = window.width, window.height
            outpath = os.path.join(out_path,output_filename.format(int(window.col_off), int(window.row_off)))
            outpath_npy = str(os.path.join(out_path,output_filename_npy.format(int(window.col_off), int(window.row_off))))
            outpath_npyblack = str(os.path.join(out_path,output_filename_npyblack.format(int(window.col_off), int(window.row_off))))


            if((int(window.width)==512) and (int(window.height)==512)):
                with rio.open(outpath, 'w', **meta) as outds:                
                    array=inds.read(window=window)
                    sum_channels= np.sum(array,axis=0)
                    equals0=(sum_channels==0).astype(np.uint8)
                    sum_percent=np.sum(equals0)/(512*512)

                    if np.sum(equals0)/(512*512) > 0.15 :
                         np.save(outpath_npyblack,array)                
                    else :
                        outds.write(array)     
                        np.save(outpath_npy,array)


                input_id=str(output_filename.format(int(window.col_off), int(window.row_off)))     
                source_id=str(os.path.join(out_path,output_filename.format(int(window.col_off), int(window.row_off))))
                coordinates2= str(coordinates.format(int(window.row_off),int(window.row_off)+int(window.width),int(window.col_off),int(window.col_off)+int(window.height)))
                percent_black=str(sum_percent)
                
                myData = [[input_id, source_id, coordinates2,percent_black]]           
                myFile = open('splits_images.csv', 'a')
                with myFile:
                    writer = csv.writer(myFile)
                    writer.writerows(myData)  


data_path = Path('imagenes')
out_path_images = '/home/crimson/fire_damage_detection_in_green_areas/LandSat-Puno/crops/'
start = timeit.default_timer()
input_filename0  = '/home/crimson/fire_damage_detection_in_green_areas/LandSat-Puno/Puno1.tif'
output_filename0 = 'Puno-2022-09-20rgbnir1{}-{}.tif'
output_filename_npy0 = 'Puno-2022-09-20rgbnir1{}-{}.npy'
output_filename_npyblack0 = 'Puno-2022-09-20rgbnir1{}-{}_b.npy'
splits_images(data_path,out_path_images,input_filename0,output_filename0,output_filename_npy0,output_filename_npyblack0)

end = timeit.default_timer()
print("elapsed time: {}".format(end-start))