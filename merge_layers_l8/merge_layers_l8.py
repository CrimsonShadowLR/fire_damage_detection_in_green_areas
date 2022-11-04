import numpy as np
import rasterio

path_img_source="./LandSat-Puno/LC09_L2SP_002071_20220918_20220920_02_T1_SR_B"

img = rasterio.open(path_img_source+"1.TIF")

lay1 = rasterio.open(path_img_source+"1.TIF").read(1)
lay2 = rasterio.open(path_img_source+"2.TIF").read(1)
lay3 = rasterio.open(path_img_source+"3.TIF").read(1)
lay4 = rasterio.open(path_img_source+"4.TIF").read(1)
lay5 = rasterio.open(path_img_source+"5.TIF").read(1)
lay6 = rasterio.open(path_img_source+"6.TIF").read(1)
lay7 = rasterio.open(path_img_source+"7.TIF").read(1)
lay10 = rasterio.open("./LandSat-Puno/LC09_L2SP_002071_20220918_20220920_02_T1_ST_B10.TIF").read(1)

meta = img.meta.copy()

meta['count'] = 8
path_img_out_tif="./LandSat-Puno/Puno1.tif"

img_out=rasterio.open(path_img_out_tif, 'w', **meta)

img_out.write(lay1, 1)
img_out.write(lay2, 2)
img_out.write(lay3, 3)
img_out.write(lay4, 4)
img_out.write(lay5, 5)
img_out.write(lay6, 6)
img_out.write(lay7, 7)
img_out.write(lay10, 8)