"adapted from Ferdinand Pineda's code"
# Ejecutar en terminal, fuera de environments conda
# Gdal me da conflictos en enviroment conda
# Instalar GDAL en el SO (Mi caso Ubuntu 22.04)

# Librerias a utilizar
import os
import glob

from pathlib import Path
import numpy as np

def merge_bands(path_img_source,path_img_out_tif):
    # R20m folder
    validator=path_img_source.split('_20m.')
    prefix=""
    cut_name=-4
    if len(validator)>1:
        prefix="_20m"
        cut_name=-8

    # Imagenes fuente
    img_bandaB = glob.glob(path_img_source+"*B02"+prefix+".jp2")
    img_bandaB = img_bandaB[0]
    img_bandaG = glob.glob(path_img_source+"*B03"+prefix+".jp2")
    img_bandaG = img_bandaG[0]
    img_bandaR = glob.glob(path_img_source+"*B04"+prefix+".jp2")
    img_bandaR = img_bandaR[0]
    img_bandaNIR = glob.glob(path_img_source+"*B8A"+prefix+".jp2")
    img_bandaNIR = img_bandaNIR[0]
    img_bandaSWIR = glob.glob(path_img_source+"*B12"+prefix+".jp2")
    img_bandaSWIR = img_bandaSWIR[0]# Obtener el nombre del archivo
    #file_name = os.path.splitext(img_bandaB)[0]
    #print(file_name)

    # Obtener el nombre del archivo sin extension
    # Para generar el tif con el nombre original
    #Se quita _B0x para que no se duplique el nombre
    file_nameB = Path(img_bandaB).stem
    print(file_nameB[:cut_name])

    # Linea para ejecuar en terminal con OS.system
    merge_bands = "gdal_merge.py -separate -o " + path_img_out_tif+file_nameB[:-4] + ".tif" + " -ot Int16 " + img_bandaB + " " + img_bandaG + " " + img_bandaR + " " + img_bandaNIR + " " + img_bandaSWIR
    #print(merge_bands)

    # Ejecutar el comando, para crear el mapa con las 4 bandas
    os.system(merge_bands)
        


# Ruta a la carpeta con las imagenes jp2
path_img_source = "/home/crimson/S2B_MSIL1C_20200612T152639_N0209_R025_T17LRL_20200612T201212/S2B_MSIL1C_20200612T152639_N0209_R025_T17LRL_20200612T201212.SAFE/GRANULE/L1C_T17LRL_A017068_20200612T153318/IMG_DATA/T17LRL_20200612T152639_"

# Ruta donde se guarda el mapa con 4 cpas, RGB y NIR
path_img_out_tif = "/home/crimson/S2B_MSIL1C_20200612T152639_N0209_R025_T17LRL_20200612T201212/"

def search_and_merge():
    # search all folders in sentinel folder
    path_images_folder='./data/images'
    folder_names = np.array(sorted(glob.glob(path_images_folder + "/")))
    print(folder_names)
    print("==================================")
    for folder_name in folder_names:
        file_folders= np.array(sorted(glob.glob(folder_name + "/")))
        print(file_folders)
        print("-------------------------------")
        for file_folder in file_folders:
            print(file_folder)
            print("**********************")
            # merge_bands(file_folder,file_folder)


if __name__ == "__main__":
    search_and_merge()