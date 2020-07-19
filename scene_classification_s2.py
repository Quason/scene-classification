from glob import glob
import os
import argparse
import shutil

import gdal
import numpy as np


def resample(src_dir, dst_dir, resolution):
    class_band_list = ['B02', 'B03', 'B04', 'B08', 'B10', 'B11', 'B12']
    mid_list1 = glob(os.path.join(src_dir, 'GRANULE', '*'))[0]
    mid_list2 = os.path.join(mid_list1, 'IMG_DATA')
    fn_list = glob(os.path.join(mid_list2, '*.jp2'))
    if resolution == 20 or resolution == 60:
        resample_alg = gdal.gdalconst.GRIORA_Average
    else:
        resample_alg = gdal.gdalconst.GRIORA_Bilinear
    for item in fn_list:
        warp_options = gdal.WarpOptions(xRes=resolution, yRes=-1*resolution,
            resampleAlg=resample_alg, format='GTiff')
        name_short = os.path.split(item)[1]
        dst_fn = os.path.join(dst_dir, name_short.replace('.jp2', '.tiff'))
        if (name_short[-7:-4]) in class_band_list:
            print(os.path.split(item)[1])
            gdal.Warp(dst_fn, item, options=warp_options)


def band_math(band_list, equation, scale=1e-4):
    for i, item in enumerate(band_list):
        ds = gdal.Open(item)
        ds_data = ds.GetRasterBand(1).ReadAsArray()
        ds_data = ds_data.astype(float) * scale
        exec('B%d=ds_data' % (i+1))
    return eval(equation)


def array2tif(array, geo_trans, proj_ref, dst_fn, type='float'):
    driver = gdal.GetDriverByName('GTiff')
    if len(np.shape(array)) == 2:
        nbands = 1
    else:
        nbands = np.shape(array)[2]
    if type == 'uint8':
        target_ds = driver.Create(
            dst_fn, np.shape(array)[1], np.shape(array)[0], nbands,
            gdal.GDT_Byte
        )
        mask_value = 255
    elif type == 'uint16':
        target_ds = driver.Create(
            dst_fn, np.shape(array)[1], np.shape(array)[0], nbands,
            gdal.GDT_UInt16
        )
        mask_value = 65535
    elif type == 'int':
        target_ds = driver.Create(
            dst_fn, np.shape(array)[1], np.shape(array)[0], nbands,
            gdal.GDT_Int16
        )
        mask_value = -999
    else:
        target_ds = driver.Create(
            dst_fn, np.shape(array)[1], np.shape(array)[0], nbands,
            gdal.GDT_Float32
        )
        mask_value = -999
    target_ds.SetGeoTransform(geo_trans)
    target_ds.SetProjection(proj_ref)
    if nbands == 1:
        target_ds.GetRasterBand(1).WriteArray(array)
        target_ds.GetRasterBand(1).SetNoDataValue(mask_value)
    else:
        for i in range(nbands):
            target_ds.GetRasterBand(i+1).WriteArray(array[:,:,i])
            target_ds.GetRasterBand(i+1).SetNoDataValue(mask_value)
    target_ds = None


def classfication(fn_list, dst_fn):
    band2 = [item for item in fn_list if 'B02' in item][0]
    band3 = [item for item in fn_list if 'B03' in item][0]
    band4 = [item for item in fn_list if 'B04' in item][0]
    band8 = [item for item in fn_list if 'B08' in item][0]
    band10 = [item for item in fn_list if 'B10' in item][0]
    band11 = [item for item in fn_list if 'B11' in item][0]
    band12 = [item for item in fn_list if 'B12' in item][0]
    red = band_math([band4], 'B1')
    classified = np.zeros(red.shape, np.uint8)
    ndsi = band_math([band3, band11], '(B1-B2)/(B1+B2)')

    b04_lim = [0.06, 0.25]
    cloud_prob1 = np.clip(red, b04_lim[0], b04_lim[1])
    cloud_prob1 = (cloud_prob1-b04_lim[0]) / (b04_lim[1]-b04_lim[0])
    ndsi_lim = [-0.24, -0.16]
    cloud_prob2 = np.clip(ndsi, ndsi_lim[0], ndsi_lim[1])
    cloud_prob2 = (cloud_prob2-ndsi_lim[0]) / (ndsi_lim[1]-ndsi_lim[0])
    cloud_prob = (cloud_prob1 * cloud_prob2 * 100).astype(np.uint8)
    cloud_prob[red<b04_lim[0]] = 0
    cloud_prob[red>b04_lim[1]] = 100
    classified[red>b04_lim[1]] = 1
    # no data
    no_data = red==0
    classified[no_data] = 1
    # snow
    nir = band_math([band8], 'B1')
    blue = band_math([band2], 'B1')
    b2r = band_math([band2, band4], 'B1/B2')
    swir2 = band_math([band12], 'B1')
    snow = (ndsi>0.2) * (nir>0.15) * (blue>0.18) * (b2r>0.85) \
        * (swir2<0.12) * (classified==0)
    classified[snow] = 1
    nir, blue, b2r, swir2 = None, None, None, None
    # vegetation
    ndvi = band_math([band8, band4], '(B1-B2)/(B1+B2)')
    nir2g = band_math([band8, band3], 'B1/B2')
    veg = (ndvi>0.36) * (nir2g>0.4) * (classified==0)
    classified[veg] = 1
    ndvi, nir2g = None, None
    # soil
    b2swir = band_math([band2, band11], 'B1/B2')
    nir2swir = band_math([band8, band11], 'B1/B2')
    soil = (b2swir<1.5) * (classified==0)
    classified[soil] = 1
    # water
    water = (ndsi>0) * (classified==0)
    classified[water] = 1
    # rock and sand
    rock_sand = (nir2swir<0.9) * (classified==0)
    classified[rock_sand] = 1
    b2swir, nir2swir = None, None
    # cirrus
    b_cirrus = band_math([band10], 'B1')
    cirrus = (b_cirrus>0.012) * (b_cirrus<0.035) * (classified==0)
    classified[cirrus] = 1
    cloud_hp = cloud_prob>65
    cloud_mp = (cloud_prob>35) * (cloud_prob<=65)
    b_cirrus = None
    # map
    class_map = np.zeros((veg.shape), np.uint8) + 7 # 默认未分类
    class_map[no_data] = 0
    class_map[veg] = 4
    class_map[soil] = 5
    class_map[rock_sand] = 5
    class_map[water] = 6
    class_map[cloud_mp] = 8
    class_map[cloud_hp] = 9
    class_map[cirrus] = 10
    class_map[snow] = 11
    ds = gdal.Open(band4)
    array2tif(
        class_map,
        ds.GetGeoTransform(),
        ds.GetProjection(),
        dst_fn,
        type='uint8'
    )


def merge(fn_list):
    ds = gdal.Open(fn_list[0])
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    band_stack = np.zeros((xsize, ysize, len(fn_list)), np.uint16)
    for i, item in enumerate(fn_list):
        ds = gdal.Open(item)
        band_stack[:,:,i] = ds.GetRasterBand(1).ReadAsArray()
    array2tif(band_stack, ds.GetGeoTransform(), ds.GetProjection(),
        os.path.join(os.path.split(fn_list[0])[0], 'stack.tif'),
        type='uint16')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='scene classification for Sentinel-2')
    parser.add_argument('--src_dir', type=str, help='original L1C data path')
    parser.add_argument('--dst_fn', type=str, default='./SC.tiff', help='export file')
    parser.add_argument('--resolution', type=int, default='', help='sensor resolution')
    args = parser.parse_args()
    dst_dir = os.path.join(os.path.split(args.dst_fn)[0], 'resample')
    os.makedirs(dst_dir, exist_ok=True)
    print('resampling ...')
    resample(args.src_dir, dst_dir, args.resolution)
    fn_list = glob(os.path.join(dst_dir, '*_B*.tiff'))
    print('classification ...')
    classfication(fn_list, args.dst_fn)
    shutil.rmtree(dst_dir)
    print('Done!')
