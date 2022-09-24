# code from https://gis.stackexchange.com/questions/222394/how-to-convert-file-shp-to-tif-using-ogr-or-python-or-gdal
# ogrinfo -so ./data/shapes/T18LZK.shp T18LZK
# gdal_rasterize -a ID -ts 512 512 -l T18LZK ./data/shapes/T18LZK.shp Result.tif
# 799980.0000,805100.0000,8494880.0000,8500000.0000
# gdal_rasterize -a ID -te  -tr  -l T18LZK T18LZK.shp Result.tif
# gdal_rasterize -l Cus-00-0 -a id -ts 512.0 512.0 -a_nodata 0.0 -te 799980.0 8494880.0 805100.0 8500000.0 -ot Byte -of GTiff C:/Users/user/Desktop/masks/Cus2020-10-06/Cus-00-0.shp C:/Users/user/Desktop/masks/Cus-2020-10-06rgbnir00-00_mask.tif

# A script to rasterise a shapefile to the same projection & pixel resolution as a reference image.
from osgeo import ogr, gdal
import subprocess

InputVector = 'T18LZK.shp'
OutputImage = 'T18LZK.tif'

RefImage = 'Cus-2020-10-06rgbnir00-0.tif'

gdalformat = 'GTiff'
datatype = gdal.GDT_Byte
burnVal = 1 #value for the output image pixels
##########################################################
# Get projection info from reference image
Image = gdal.Open(RefImage, gdal.GA_ReadOnly)

# Open Shapefile
Shapefile = ogr.Open(InputVector, 1)
Shapefile_layer = Shapefile.GetLayer()
print(Shapefile_layer.GetExtent())
print(Shapefile_layer.GetFeature(1))

# Rasterise
print(Image.RasterXSize)
print(Image.RasterYSize)
print("Rasterising shapefile...")
Output = gdal.GetDriverByName(gdalformat).Create(OutputImage, Image.RasterXSize, Image.RasterYSize, 1, datatype)
print(type(Output))
print(type(Image))
Output.SetProjection(Image.GetProjectionRef())
Output.SetGeoTransform(Image.GetGeoTransform()) 
print(Image.GetProjectionRef())
print(Image.GetGeoTransform())
print(Output.GetProjectionRef())
print(Output.GetGeoTransform())

# Write data to band 1
Band = Output.GetRasterBand(1)
Band_array = Output.GetRasterBand(1).ReadAsArray()
# Band.SetNoDataValue(0)
gdal.RasterizeLayer(Output, [1], Shapefile_layer, burn_values=[burnVal])
# print(Output.count)
print('min ', Band_array.min(), ' max ', Band_array.max(), Band_array.mean())

# Close datasets
Band = None
Output = None
Image = None
Shapefile = None

# Build image overviews
# subprocess.call("gdaladdo --config COMPRESS_OVERVIEW DEFLATE "+OutputImage+" 2 4 8 16 32 64", shell=True)
print("Done.")