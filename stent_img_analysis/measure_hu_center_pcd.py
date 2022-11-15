# This script will run through the images selected, crop the images, and then measure the size of the stent wires and
# the measured diameter is

import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from glob import glob
import cv2
from find_nearest import find_nearest_index
import draw_profile as dp
from general_functions import crop_array
from mask_functions import phantom_ROIs, click_image, circular_mask

# directory = r'C:\Users\drich\OneDrive - University of Victoria\Research\LDA Data'
directory = r'D:\OneDrive - University of Victoria\Research\LDA Data'

folder = '22_10_11_CT_stents_heli'
sub = 'pink_mid'
append = '_HR'
# append = ''
mask_radius = 9  # 9, 7

path = os.path.join(directory, folder, sub, 'Norm CT', f'CT_FDK{append}.npy')

# good_slices = [5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]  # pink_mid
# good_slices = [3, 5, 6, 11, 13, 17, 19, 20, 21]  # purple_mid
# good_slices = [3, 5, 6, 10, 15, 16, 17, 20, 21, 22]  # red_mid

# good_slices = np.arange(4, 18)  # Helical HR
good_slices = [0, 1, 2, 3, 4, 5, 6, 7]  # Helical LR
num_slices = len(good_slices)  # or len(good_slices)

# Open the first good image in order to find the stent and crop
data = np.load(path)[-1]  # Open data, last bin
im = data[good_slices[3]]
coords_path = os.path.join(directory, folder, sub, f'corner_coords{append}.npy')
if os.path.exists(coords_path):
    coords = np.load(coords_path)
else:
    coords = crop_array(im)  # Crop
    np.save(coords_path, coords)

im = im[coords[0]:coords[1], coords[2]:coords[3]]

# The center coordinate
center_coord_path = os.path.join(directory, folder, sub, f'center_coords{append}.npy')
if os.path.exists(center_coord_path):
    center_coords = np.load(center_coord_path)
else:
    center_coords = np.squeeze(click_image(data[good_slices[3]]))
    np.save(center_coord_path, center_coords)

# Collect all the widths and diameters
hu_mean = np.zeros(num_slices)
hu_var = np.zeros(num_slices)
hu_pixels = []

# Crop the whole data array
data = data[:, coords[0]:coords[1], coords[2]:coords[3]]

# The image mask
mask = circular_mask(center_coords, radius=mask_radius, img_dim=np.shape(data[good_slices[0]]))

# Go through the appropriate files and calculate the center, find an edge point and calculate width and diameter
for idx, im1 in enumerate(data[good_slices]):

    hu = im1 * mask
    hu = hu[~np.isnan(hu)]
    hu_pixels.append(hu)

    # hu_mean[idx] = np.nanmean(im1*mask)
    # hu_var[idx] = np.nanvar(im1*mask)

    # fig = plt.figure()
    # plt.imshow(im1)
    # plt.imshow(mask, alpha=0.8)
    # plt.pause(1)
    # plt.close()

# print(hu_mean)
# print(np.mean(hu_mean))
# print()
# print(np.sqrt(hu_var))
# print(np.mean(np.sqrt(hu_var)))

# np.save(os.path.join(directory, folder, sub, f'HU_mean{append}.npy'), hu_mean)
# np.save(os.path.join(directory, folder, sub, f'HU_variance{append}.npy'), hu_var)
print(np.mean(hu_pixels))
print(np.std(hu_pixels))

np.save(os.path.join(directory, folder, sub, f'HU_pixels{append}.npy'), hu_pixels)

# low_corr_mean[z - gd_sls1[0]] = np.nanmean(corr_temp)
# low_corr_std[z - gd_sls1[0]] = np.nanvar(corr_temp)
# low_corr_mean = np.mean(low_corr_mean)
# low_corr_std = np.sqrt(np.mean(low_corr_std))
