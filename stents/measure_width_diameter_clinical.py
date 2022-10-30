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
folder = '22_09_20_CT_stents'
sub = 'red'

good_slices = np.arange(90, 100)  # BC Cancer

# good_slices = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]  # Red BC Cancer
# good_slices = [24, 25, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46]  # Purple BC Cancer
# good_slices = [24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]  # Pink BC Cancer

# good_slices = [121, 127, 128, 130, 131, 133, 134]  # Red Initio
# good_slices = [133, 139, 141, 142, 143, 144]  # Purple Initio
# good_slices = [128, 30, 131, 133, 134, 135, 136, 137]  # Pink Initio


num_slices = len(good_slices)
num_interp_pts = 100  # Number of points in the interpolation of the profile lines from the center to a point outside
degree = 5  # Degree between each of the points outside the circle (number of profiles)
num_profiles = int(360/degree)


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
edge_pt_path = os.path.join(directory, folder, '10cm_phantom', sub, 'edge_pt_coords.npy')
if os.path.exists(edge_pt_path):
    ep = np.load(edge_pt_path)
else:
    ep = np.squeeze(click_image(im))  # Edge point outside the circle
    np.save(edge_pt_path, ep)

# The center coordinate
center_coord_path = os.path.join(directory, folder, '10cm_phantom', sub, f'center_coords.npy')
if os.path.exists(center_coord_path):
    center_coords = np.load(center_coord_path)
else:
    center_coords = np.squeeze(click_image(im))
    np.save(center_coord_path, center_coords)

# Collect all the widths and diameters
widths = []
radii = []
out_radii = []

px_sz = data['PixelSpacing'].value[0]  # Pixel size

# Go through the appropriate files and calculate the center, find an edge point and calculate width and diameter
# for idx, file in enumerate(files[good_slices[0]:good_slices[1]]):
#     val = idx + good_slices[0]
for idx, val in enumerate(good_slices):
    file = files[val]

    data = pyd.dcmread(file)

    im = data.pixel_array  # Get just the pixel array

    im1 = im[coords[0]:coords[1], coords[2]:coords[3]]
    im2 = np.copy(im1) * data.RescaleSlope + data.RescaleIntercept
    # im1 = ((im1 - im1.min()) * (1/(im1.max() - im1.min()) * 255)).astype('uint8')
    # im1 = cv2.medianBlur(im1, 5)
    # circle = cv2.HoughCircles(im1, cv2.HOUGH_GRADIENT, 1, 20, param1=15, param2=10, minRadius=0, maxRadius=0)

    # fig = plt.figure()

    # Collect the center point
    # circle = circle[0]
    # center = (circle[0, 0], circle[0, 1])
    #
    # fig, ax = plt.subplots(1, 1)
    # plt.imshow(im2, vmin=-500, vmax=2000, cmap='gray')
    # plt.scatter(30, 33)
    # circ = plt.Circle((30, 33), radius=8, fill=False, edgecolor='red')
    # ax.add_artist(circ)
    # plt.title(f'{val}\n{center}')
    # plt.show()
    # plt.pause(1)
    # plt.close()

    # Collect points equidistant from the center around the outside of the cirlce
    edge_pts = dp.rotate_profile(center_coords, degree, ep)

    # plt.title(val)
    # plt.imshow(im1)
    # plt.scatter(center[0], center[1], color='red')
    for pidx, point in enumerate(edge_pts):

        temp_diameter, temp_width, temp_out = dp.find_dist_with_fwhm(center_coords, point, im2,
                                                                     num_new_pts=num_interp_pts,
                                                                     vxl_sz=px_sz)

        widths.append(temp_width)
        radii.append(temp_diameter)
        out_radii.append(temp_out)

        # plt.scatter(point[0], point[1], color='black')

    # plt.show()
#
#
print(rf'Wire Width: {np.mean(widths)} $\pm$ {np.std(widths)} mm')
print(rf'Diameter: {np.mean(radii)*2} $\pm$ {np.std(radii)*2} mm')
print(rf'Outer Diameter: {np.mean(out_radii)*2} $\pm$ {np.std(out_radii)*2} mm')

np.save(os.path.join(directory, folder, '10cm_phantom', sub, 'widths.npy'), widths)
np.save(os.path.join(directory, folder, '10cm_phantom', sub, 'radii.npy'), radii)
np.save(os.path.join(directory, folder, '10cm_phantom', sub, 'outer_radii.npy'), out_radii)
