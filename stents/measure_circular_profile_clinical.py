import os
import numpy as np
import pydicom as pyd
import matplotlib.pyplot as plt
from natsort import natural_keys
from glob import glob
from general_functions import crop_array
from scipy.ndimage import map_coordinates
from mask_functions import click_image

# Set the location of the files you want to analyze
# directory = r'C:\Users\drich\OneDrive - University of Victoria\Research\Clinical CT'
directory = r'D:\OneDrive - University of Victoria\Research\Clinical CT'

folder = '22_09_20_CT_stents'
sub = 'purple'

# good_slices = [121, 127, 128, 130, 131, 133, 134]  # Red Initio
good_slices = [133, 139, 141, 142, 143, 144]  # Purple Initio
# good_slices = [128, 30, 131, 133, 134, 135, 136, 137]  # Pink Initio

# good_slices = [24, 25, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46]  # Purple
# good_slices = [24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]  # Pink
# good_slices = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]  # Red

path = os.path.join(directory, folder, '10cm_phantom', sub, 'Data')
files = glob(os.path.join(path, '*.dcm'))
files.sort(key=natural_keys)

# Open the 21st image in order to find the stent and crop
data = pyd.dcmread(files[good_slices[0]])
im = data.pixel_array * data.RescaleSlope + data.RescaleIntercept
px_sz = data['PixelSpacing'].value[0]  # Pixel size
coords_path = os.path.join(directory, folder, '10cm_phantom', sub, 'corner_coords.npy')
if os.path.exists(coords_path):
    coords = np.load(coords_path)
else:
    coords = crop_array(im)  # Crop
    np.save(coords_path, coords)

im = im[coords[0]:coords[1], coords[2]:coords[3]]

# The profile coordinates
prof_coords_path = os.path.join(directory, folder, '10cm_phantom', sub, f'profile_coords.npy')
if os.path.exists(prof_coords_path):
    prof_coords = np.load(prof_coords_path)
else:
    prof_coords = np.squeeze(click_image(im))
    np.save(prof_coords_path, prof_coords)
prof_coords = np.array(prof_coords, dtype='int')

# The distances between subsequent pairs of coordinates
dist_prof = np.zeros(len(prof_coords)+1)
for i in range(len(prof_coords[:-1])):
    dist_prof[i+1] = np.sqrt((prof_coords[i][0] - prof_coords[i+1][0])**2 + (prof_coords[i][1] - prof_coords[i+1][1])**2)
dist_prof[-1] = np.sqrt((prof_coords[0][0] - prof_coords[-1][0])**2 + (prof_coords[0][1] - prof_coords[-1][1])**2)

num_pts_seg = 5
xpts = np.linspace(prof_coords[0][0], prof_coords[1][0], int(num_pts_seg*dist_prof[0]))
ypts = np.linspace(prof_coords[0][1], prof_coords[1][1], int(num_pts_seg*dist_prof[0]))
for i in np.arange(1, len(prof_coords[1:-1])):
    xpts = np.concatenate((xpts, np.linspace(prof_coords[i][0], prof_coords[i+1][0], int(num_pts_seg*dist_prof[i]))))
    ypts = np.concatenate((ypts, np.linspace(prof_coords[i][1], prof_coords[i + 1][1], int(num_pts_seg*dist_prof[i]))))
xpts = np.concatenate((xpts, np.linspace(prof_coords[-1][0], prof_coords[0][0], int(num_pts_seg*dist_prof[-1]))))
ypts = np.concatenate((ypts, np.linspace(prof_coords[-1][1], prof_coords[0][1], int(num_pts_seg*dist_prof[-1]))))

dist = np.zeros(len(xpts))
for i in range(len(xpts[:-1])):
    dist[i+1] = dist[i] + np.sqrt((xpts[i] - xpts[i+1])**2 + (ypts[i] - ypts[i+1])**2)
dist = dist * px_sz  # Convert to mm from px

# Go through the appropriate files and calculate the center, find an edge point and calculate width and diameter
for idx, val in enumerate(good_slices):
    file = files[val]

    data = pyd.dcmread(file)

    im = data.pixel_array * data.RescaleSlope + data.RescaleIntercept  # Get just the pixel array

    im1 = im[coords[0]:coords[1], coords[2]:coords[3]]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(im1, vmin=-500, vmax=800, cmap='gray')
    ax[0].axis('off')

    # Plot linear segments between each set of points to make sure we have
    for c in range(len(prof_coords[:-1])):
        ax[0].plot((prof_coords[c][0], prof_coords[c+1][0]), (prof_coords[c][1], prof_coords[c+1][1]), color='red')

    # Extract the values along the line, using cubic interpolation
    intp_prof = map_coordinates(im1, np.vstack((ypts, xpts)))

    ax[1].plot(dist, intp_prof, color='blue')
    ax[1].set_xlabel('Distance (mm)')
    ax[1].set_ylabel('HU')

    ax[0].annotate('A', (7, 192), xycoords='axes points', color='white', fontsize=14)
    ax[1].annotate('B', (7, 203), xycoords='axes points', color='black', fontsize=14)

    fig.savefig(os.path.join(directory, folder, '10cm_phantom', sub, 'fig', f'fig_{idx}.png'), dpi=500)

    break
