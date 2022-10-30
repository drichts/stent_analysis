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
from mask_functions import click_image

# directory = r'C:\Users\drich\OneDrive - University of Victoria\Research\LDA Data'
directory = r'D:\OneDrive - University of Victoria\Research\LDA Data'

folder = '22_10_11_CT_stents_heli'
sub = 'pink_mid'
append = ''

path = os.path.join(directory, folder, sub, 'Norm CT', f'CT_FDK{append}.npy')

# good_slices = [3, 5, 6, 10, 15, 16, 17, 20, 21, 22]  # red_mid
# good_slices = [3, 5, 6, 11, 13, 17, 19, 20, 21]  # purple_mid
# good_slices = [5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]  # pink_mid

# good_slices = np.arange(4, 16)
good_slices = np.arange(8)
num_slices = len(good_slices)  # or len(good_slices)
num_interp_pts = 100  # Number of points in the interpolation of the profile lines from the center to a point outside
degree = 5  # Degree between each of the points outside the circle (number of profiles)
num_profiles = int(360/degree)


# Open the first good image in order to find the stent and crop
data = np.load(path)[-1]  # Open data, last bin
im = data[good_slices[2]]
coords_path = os.path.join(directory, folder, sub, f'corner_coords{append}.npy')
if os.path.exists(coords_path):
    coords = np.load(coords_path)
else:
    coords = crop_array(im)  # Crop
    np.save(coords_path, coords)

im = im[coords[0]:coords[1], coords[2]:coords[3]]

edge_pt_path = os.path.join(directory, folder, sub, f'edge_pt_coords{append}.npy')
if os.path.exists(edge_pt_path):
    ep = np.load(edge_pt_path)
else:
    ep = np.squeeze(click_image(im))  # Edge point outside the circle
    np.save(edge_pt_path, ep)

# The center coordinate
center_coord_path = os.path.join(directory, folder, sub, f'center_coords{append}.npy')
if os.path.exists(center_coord_path):
    center_coords = np.load(center_coord_path)
else:
    center_coords = np.squeeze(click_image(im))
    np.save(center_coord_path, center_coords)

# Collect all the widths and diameters
widths = []
radii = []
out_radii = []

px_sz = 105 / np.shape(data[0])[0]  # Pixel size (recon size divided by number of pixels)

# Crop the whole data array
data = data[:, coords[0]:coords[1], coords[2]:coords[3]]


# Go through the appropriate files and calculate the center, find an edge point and calculate width and diameter
for idx, im1 in enumerate(data[good_slices]):

    val = good_slices[idx]

    # Collect points equidistant from the center around the outside of the cirlce
    edge_pts = dp.rotate_profile(center_coords, degree, ep)

    # fig = plt.figure()
    # plt.title(val)
    # plt.imshow(im1)
    # plt.scatter(center_coords[0], center_coords[1], color='red')
    for pidx, point in enumerate(edge_pts):
        temp_diam, temp_width, temp_out = dp.find_dist_with_fwhm(center_coords, point, im1, num_new_pts=num_interp_pts,
                                                                 vxl_sz=px_sz)

        if 3 > temp_diam > 0:
            widths.append(temp_width)
            radii.append(temp_diam)
            out_radii.append(temp_out)

        # plt.scatter(point[0], point[1], color='black')

    # plt.show()
    # plt.pause(1)

print(rf'Diameter: {np.mean(radii)*2} $\pm$ {np.std(radii)*2} mm')
print(rf'Wire Width: {np.mean(widths)} $\pm$ {np.std(widths)} mm')
print(rf'Outer Diameter: {np.mean(out_radii)*2} $\pm$ {np.std(out_radii)*2} HU')

# np.save(os.path.join(directory, folder, sub, f'radii{append}.npy'), radii)
# np.save(os.path.join(directory, folder, sub, f'widths{append}.npy'), widths)
# np.save(os.path.join(directory, folder, sub, f'outer_diam{append}'), out_radii)
