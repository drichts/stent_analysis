# This script will run through the images selected, crop the images, and then measure the size of the stent wires and
# the measured diameter is

import os
import numpy as np
import pydicom as pyd
from natsort import natural_keys
from glob import glob
from general_functions import crop_array
from mask_functions import phantom_ROIs, click_image, circular_mask
import matplotlib.pyplot as plt

# directory = r'C:\Users\drich\OneDrive - University of Victoria\Research\Clinical CT'
directory = r'D:\OneDrive - University of Victoria\Research\Clinical CT'

folder = '22_10_19_CT_stents'
sub = 'red_boneplus'
mask_radius = 6  # 7, 6

good_slices = np.arange(80, 90)  # BC Cancer

# good_slices = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]  # Red
# good_slices = [24, 25, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46]  # Purple
# good_slices = [24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]  # Pink

# good_slices = [121, 127, 128, 130, 131, 133, 134]  # Red Initio
# good_slices = [133, 139, 141, 142, 143, 144]  # Purple Initio
# good_slices = [128, 130, 131, 133, 134, 135, 136, 137]  # Pink Initio

# good_slices = [133]

num_slices = len(good_slices)


path = os.path.join(directory, folder, '10cm_phantom', sub, 'Data')
files = glob(os.path.join(path, '*.dcm'))
files.sort(key=natural_keys)

# Open the 21st image in order to find the stent and crop
data = pyd.dcmread(files[good_slices[0]])
im = data.pixel_array
coords_path = os.path.join(directory, folder, '10cm_phantom', sub, 'corner_coords.npy')
if os.path.exists(coords_path):
    coords = np.load(coords_path)
else:
    coords = crop_array(im)  # Crop
    np.save(coords_path, coords)

im = im[coords[0]:coords[1], coords[2]:coords[3]]

# Collect all the widths and diameters
# hu_mean = np.zeros(num_slices)
# hu_var = np.zeros(num_slices)
hu_pixels = []

# The center coordinate
center_coord_path = os.path.join(directory, folder, '10cm_phantom', sub, 'center_coords.npy')
if os.path.exists(center_coord_path):
    center_coords = np.load(center_coord_path)
else:
    center_coords = np.squeeze(click_image(im))
    np.save(center_coord_path, center_coords)
#
#
# # The image mask
mask = circular_mask(center_coords, radius=mask_radius, img_dim=np.shape(im))  # BC Cancer

# Go through the appropriate files and calculate the center, find an edge point and calculate width and diameter
for idx, val in enumerate(good_slices):
    file = files[val]

    data = pyd.dcmread(file)

    im1 = data.pixel_array * data.RescaleSlope + data.RescaleIntercept  # Get just the pixel array

    im1 = im1[coords[0]:coords[1], coords[2]:coords[3]]

    # # # For the initio data
    # mask_path = os.path.join(directory, folder, '10cm_phantom', sub, f'center_mask_slice{val}.npy')
    # if os.path.exists(mask_path):
    #     mask = np.load(mask_path)
    # else:
    #     mask = phantom_ROIs(im1, radius=mask_radius)[0]
    #     np.save(mask_path, mask)

    fig = plt.figure()
    plt.imshow(im1)
    plt.imshow(mask, alpha=0.8)
    plt.pause(1)
    plt.close()

    # hu_mean[idx] = np.nanmean(im1*mask)
    # hu_var[idx] = np.nanvar(im1*mask)

    hu = im1 * mask
    hu = hu[~np.isnan(hu)]
    hu_pixels.append(hu)

# print(hu_mean)
# print()
# print(np.sqrt(hu_var))
print(np.mean(hu_pixels))
print(np.std(hu_pixels))
# if not os.path.exists(mask_path):
#     np.save(mask_path, mask)
np.save(os.path.join(directory, folder, '10cm_phantom', sub, f'HU_pixels.npy'), hu_pixels)
# np.save(os.path.join(directory, folder, '10cm_phantom', sub, f'HU_mean.npy'), hu_mean)
# np.save(os.path.join(directory, folder, '10cm_phantom', sub, f'HU_variance.npy'), hu_var)

# low_corr_mean[z - gd_sls1[0]] = np.nanmean(corr_temp)
# low_corr_std[z - gd_sls1[0]] = np.nanvar(corr_temp)
# low_corr_mean = np.mean(low_corr_mean)
# low_corr_std = np.sqrt(np.mean(low_corr_std))
