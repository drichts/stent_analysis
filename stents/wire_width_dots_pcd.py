# This script will draw the selected profiles and calculate the wire width

import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from glob import glob
import cv2
from find_nearest import find_nearest_index
import draw_profile as dp
from general_functions import crop_array
from mask_functions import click_image

# directory = r'C:\Users\drich\OneDrive - University of Victoria\Research\LDA Data'
directory = r'D:\OneDrive - University of Victoria\Research\LDA Data'

folder = '22_10_11_CT_stents_heli'
sub = 'purple_mid'
append = '_HR'

path = os.path.join(directory, folder, sub, 'Norm CT', f'CT_FDK{append}.npy')

good_slices = np.arange(0, 24)
# good_slices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
# good_slices = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19]  # red High
# good_slices = np.arange(8)
num_slices = len(good_slices)  # or len(good_slices)
num_interp_pts = 100  # Number of points in the interpolation of the profile lines from the center to a point outside

# Open the first good image in order to find the stent and crop
data = np.load(path)[-1]  # Open data, last bin
im = data[good_slices[0]]
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
    center_coords = np.squeeze(click_image(im))
    np.save(center_coord_path, center_coords)

# Collect all the widths
widths = []
radii = []
out_radii = []

px_sz = 105 / np.shape(data[0])[0]  # Pixel size (recon size divided by number of pixels)

# Crop the whole data array
data = data[:, coords[0]:coords[1], coords[2]:coords[3]]

# Go through the appropriate files and calculate the center, find an edge point and calculate width and diameter
for idx, im1 in enumerate(data[good_slices]):

    val = good_slices[idx]

    # Collect points along a path from the center through a wire dot
    width_pts_path = os.path.join(directory, folder, sub, f'width_coords{append}_slice_{val}.npy')
    if os.path.exists(width_pts_path):
        width_pts = np.load(width_pts_path)
    else:
        width_pts = np.squeeze(click_image(im1))
        np.save(width_pts_path, width_pts)

    fig, ax = plt.subplots(1, 1)
    ax.set_title(val)

    ax.imshow(im1)
    ax.scatter(center_coords[0], center_coords[1], color='red')

    for pidx, point in enumerate(width_pts):

        temp_diam, temp_width, temp_out = dp.find_dist_with_fwhm(center_coords, point, im1, num_new_pts=num_interp_pts,
                                                                vxl_sz=px_sz)
        widths.append(temp_width)
        radii.append(temp_diam)
        out_radii.append(temp_out)

        point1 = dp.extend_point(center_coords, point)
        ax.plot((center_coords[0], point1[0]), (center_coords[1], point1[1]), color='black')
    circ = plt.Circle((40.5, 46), radius=12, fill=False, edgecolor='red')
    ax.add_artist(circ)

    # plt.show()
    # plt.pause(1)

print(rf'Wire Width: {np.mean(widths)} $\pm$ {np.std(widths)} mm')
print(rf'Inner Diameter: {np.mean(radii)*2} $\pm$ {np.std(radii)*2} mm')
print(rf'Outer Diameter: {np.mean(out_radii)*2} $\pm$ {np.std(out_radii)*2} mm')

np.save(os.path.join(directory, folder, sub, f'widths_dots{append}.npy'), widths)
np.save(os.path.join(directory, folder, sub, f'radii_dots{append}.npy'), radii)
np.save(os.path.join(directory, folder, sub, f'outer_radii_dots{append}.npy'), out_radii)

