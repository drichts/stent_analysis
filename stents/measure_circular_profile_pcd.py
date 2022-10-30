import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from general_functions import crop_array
from mask_functions import click_image

# directory = r'C:\Users\drich\OneDrive - University of Victoria\Research\LDA Data'
directory = r'D:\OneDrive - University of Victoria\Research\LDA Data'

folder = '22_09_07_CT_stents'
sub = 'purple_mid'
append = ''

path = os.path.join(directory, folder, sub, 'Norm CT', f'CT_FDK{append}.npy')

# good_slices = [5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]  # pink_mid
good_slices = [3, 5, 6, 11, 13, 17, 19, 20, 21]  # purple_mid
# good_slices = [3, 5, 6, 10, 15, 16, 17, 20, 21, 22]  # red_mid

# good_slices = [0, 1, 2, 3, 4, 5, 6]
num_slices = len(good_slices)  # or len(good_slices)

# Open the first good image in order to find the stent and crop
data = np.load(path)[-1]  # Open data, last bin
im = data[good_slices[3]]

px_sz = 105 / np.shape(im)[0]  # Find the pixel size of the images in mm

coords_path = os.path.join(directory, folder, sub, f'corner_coords{append}.npy')
if os.path.exists(coords_path):
    coords = np.load(coords_path)
else:
    coords = crop_array(im)  # Crop
    np.save(coords_path, coords)

# Crop the whole data array
data = data[:, coords[0]:coords[1], coords[2]:coords[3]]

# The profile coordinates
prof_coords_path = os.path.join(directory, folder, sub, f'profile_coords{append}.npy')
if os.path.exists(prof_coords_path):
    prof_coords = np.load(prof_coords_path)
else:
    prof_coords = np.squeeze(click_image(data[good_slices[3]]))
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

# # The distances between subsequent pairs of coordinates
# xpts_reg = np.zeros(len(prof_coords))
# for i in range(len(prof_coords[:-1])):
#     xpts_reg[i+1] = xpts_reg[i] + np.sqrt((prof_coords[i][0] - prof_coords[i+1][0])**2 + (prof_coords[i][1] - prof_coords[i+1][1])**2)
# xpts_reg = xpts_reg * px_sz  # Convert to mm from px
# xpts_reg_int = np.linspace(0, xpts_reg[-1], 100)

# Go through the appropriate files and calculate the center, find an edge point and calculate width and diameter
for idx, im1 in enumerate(data[good_slices]):

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].axis('off')
    ax[0].imshow(im1, vmin=-500, vmax=800, cmap='gray')

    # Plot linear segments between each set of points to make sure we have
    for c in range(len(prof_coords[:-1])):
        ax[0].plot((prof_coords[c][0], prof_coords[c+1][0]), (prof_coords[c][1], prof_coords[c+1][1]), color='red')

    # ypts_reg = np.zeros(len(xpts_reg))
    # for pi, p in enumerate(prof_coords):
    #     ypts_reg[pi] = im1[p[1], p[0]]
    #
    # f = interp1d(xpts_reg, ypts_reg, kind='cubic')

    # Extract the values along the line, using cubic interpolation
    intp_prof = map_coordinates(im1, np.vstack((ypts, xpts)))

    # int_im_vals = np.zeros(len(ypts))
    # for r in range(len(xpts)):
    #     int_im_vals[r] = im1[int(round(ypts[r])), int(round(xpts[r]))]

    # ax[1].scatter(xpts_reg, ypts_reg, color='red')
    # ax[1].plot(xpts_reg_int, f(xpts_reg_int), color='red')
    # ax[1].scatter(dist, int_im_vals, color='green')
    ax[1].plot(dist, intp_prof, color='blue')
    ax[1].set_xlabel('Distance (mm)')
    ax[1].set_ylabel('HU')

    ax[0].annotate('A', (7, 203), xycoords='axes points', color='white', fontsize=14)
    ax[1].annotate('B', (7, 203), xycoords='axes points', color='black', fontsize=14)

    fig.savefig(os.path.join(directory, folder, sub, 'fig', f'fig_{idx}{append}.png'), dpi=500)

