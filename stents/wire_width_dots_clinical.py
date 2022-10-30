# This script will run through the images selected, crop the images, and then measure the size of the stent wires and
# the measured diameter is

import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom as pyd
from glob import glob
from natsort import natural_keys
import cv2
from find_nearest import find_nearest_index
import draw_profile as dp
from general_functions import crop_array
from mask_functions import click_image

# Set the location of the files you want to analyze
# directory = r'C:\Users\drich\OneDrive - University of Victoria\Research\Clinical CT'
directory = r'D:\OneDrive - University of Victoria\Research\Clinical CT'

folder = '22_10_19_CT_stents'
sub = 'purple_boneplus'

# good_slices = np.arange(80, 104)  # BC Cancer
good_slices = [80, 82, 83, 84, 85, 87, 88, 89, 91, 92, 93, 94, 95, 96, 97, 98, 101, 102, 103]
# good_slices = [24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]  # Pink BC Cancer
# good_slices = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]  # Red BC Cancer

# good_slices = np.arange(130, 145)
# good_slices = np.arange(119, 135)   # Red Initio
# good_slices = [125, 126, 127, 128, 129, 130, 131, 132, 133]  # Purple Initio

# good_slices = [128, 130, 131, 133, 134, 135, 136, 137]  # Pink Initio

num_slices = len(good_slices)
num_interp_pts = 100  # Number of points in the interpolation of the profile lines from the center to a point outside

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

# # The center coordinate
# center_coord_path = os.path.join(directory, folder, '10cm_phantom', sub, f'center_coords.npy')
# if os.path.exists(center_coord_path):
#     center_coords = np.load(center_coord_path)
# else:
#     center_coords = np.squeeze(click_image(im))
#     np.save(center_coord_path, center_coords)


# Collect all the widths
widths = []
radii = []
out_radii = []

px_sz = data['PixelSpacing'].value[0]  # Pixel size

for idx, val in enumerate(good_slices):
    file = files[val]

    data = pyd.dcmread(file)

    im = data.pixel_array  # Get just the pixel array

    im1 = im[coords[0]:coords[1], coords[2]:coords[3]]
    im2 = np.copy(im1) * data.RescaleSlope + data.RescaleIntercept
    im1 = ((im1 - im1.min()) * (1/(im1.max() - im1.min()) * 255)).astype('uint8')
    im1 = cv2.medianBlur(im1, 5)
    circle = cv2.HoughCircles(im1, cv2.HOUGH_GRADIENT, 1, 20, param1=15, param2=10, minRadius=0, maxRadius=0)

    # # Collect the center point
    circle = circle[0]
    center = (circle[0, 0], circle[0, 1])

    center_coords_path = os.path.join(directory, folder, '10cm_phantom', sub, f'center_coords_slice{val}.npy')
    center_coords = np.load(center_coords_path)
    # np.save(center_coords_path, center)
    # print(val, center)
    # if os.path.exists(center_coords_path):
    #     center_coords = np.load(center_coords_path)
        # print(val, center_coords)
        # fig, ax = plt.subplots(1, 1)
        # circ = plt.Circle(center_coords, radius=10, fill=False, edgecolor='red')
        # ax.add_artist(circ)
        # ax.imshow(im2, cmap='gray', vmin=-500, vmax=1500)
        # ax.set_title(val)
        # plt.show()
        # plt.pause(2)
        # plt.close()
    # else:
        # fig, ax = plt.subplots(1, 1)
        # center_coords = (30.5, 26.5)
        # np.save(center_coords_path, center_coords)
        # circ = plt.Circle(center_coords, radius=8, fill=False, edgecolor='red')
        # ax.add_artist(circ)
        # ax.imshow(im2, cmap='gray', vmin=-500, vmax=1500)
        # ax.set_title(val)
        # plt.show()
        # plt.pause(3)
        # plt.close()


    # Collect points along a path from the center through a wire dot
    width_pts_path = os.path.join(directory, folder, '10cm_phantom', sub, f'width_coords_slice_{val}.npy')
    if os.path.exists(width_pts_path):
        width_pts = np.load(width_pts_path)
    else:
        width_pts = np.squeeze(click_image(im2))
        np.save(width_pts_path, width_pts)

    fig, ax = plt.subplots(1, 1)
    ax.set_title(val)
    ax.imshow(im2)
    ax.scatter(center_coords[0], center_coords[1], color='red')
    ax.scatter(center_coords[0], center_coords[1], color='red')
    # print(val, center_coords)
    for pidx, point in enumerate(width_pts):

        temp_diameter, temp_width, temp_out = dp.find_dist_with_fwhm(center, point, im2, num_new_pts=num_interp_pts,
                                                                    vxl_sz=px_sz)
        # print(val, temp_width, temp_diameter)
        widths.append(temp_width)
        radii.append(temp_diameter)
        out_radii.append(temp_out)
        point1 = dp.extend_point(center_coords, point)
        plt.plot((center_coords[0], point1[0]), (center_coords[1], point1[1]), color='black')
    circ = plt.Circle(center_coords, radius=8, fill=False, edgecolor='red')
    ax.add_artist(circ)



print(rf'Wire Width: {np.mean(widths)} $\pm$ {np.std(widths)} mm')
print(rf'Diameter: {np.mean(radii)*2} $\pm$ {np.std(radii)*2} mm')
print(rf'Outer Diameter: {np.mean(out_radii)*2} $\pm$ {np.std(out_radii)*2} mm')

np.save(os.path.join(directory, folder, '10cm_phantom', sub, 'widths_dots.npy'), widths)
np.save(os.path.join(directory, folder, '10cm_phantom', sub, 'radii_dots.npy'), radii)
np.save(os.path.join(directory, folder, '10cm_phantom', sub, 'outer_radii_dots.npy'), out_radii)
