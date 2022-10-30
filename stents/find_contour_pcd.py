import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import interp1d, CubicSpline
from general_functions import crop_array
from mask_functions import click_image

# directory = r'C:\Users\drich\OneDrive - University of Victoria\Research\LDA Data'
directory = r'D:\OneDrive - University of Victoria\Research\LDA Data'

folder = '22_09_07_CT_stents'
sub = 'purple_mid'
append = '_initio_8'

path = os.path.join(directory, folder, sub, 'Norm CT', f'CT_FDK{append}.npy')

# good_slices = [5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]  # pink_mid
# good_slices = [3, 5, 6, 11, 13, 17, 19]  # purple_mid
# good_slices = [3, 5, 6, 10, 15, 16, 17, 20, 21, 22]  # red_mid

good_slices = [0, 1, 2, 3, 4, 5, 6, 7]
num_slices = len(good_slices)  # or len(good_slices)

# Open the first good image in order to find the stent and crop
data = np.load(path)[-1]  # Open data, last bin

px_sz = 105 / np.shape(data[good_slices[0]])[0]

# Open the first good image in order to find the stent and crop
data = np.load(path)[-1]  # Open data, last bin
im = data[good_slices[2]]
coords_path = os.path.join(directory, folder, sub, f'corner_coords{append}.npy')
if os.path.exists(coords_path):
    coords = np.load(coords_path)
else:
    coords = crop_array(im)  # Crop
    np.save(coords_path, coords)

# Crop the whole data array
data = data[:, coords[0]:coords[1], coords[2]:coords[3]]

# The center coordinate
center_coord_path = os.path.join(directory, folder, sub, f'center_coords{append}.npy')
if os.path.exists(center_coord_path):
    center_coords = np.load(center_coord_path)
else:
    center_coords = np.squeeze(click_image(im))
    np.save(center_coord_path, center_coords)

# Go through the appropriate files and calculate the center, find an edge point and calculate width and diameter
for idx, im1 in enumerate(data[good_slices]):

    im2 = np.copy(im1)
    im1 = ((im1 - im1.min()) * (1 / (im1.max() - im1.min()) * 255)).astype('uint8')
    im1 = cv2.medianBlur(im1, 5)
    # circle = cv2.HoughCircles(im1, cv2.HOUGH_GRADIENT, 1, 20, param1=15, param2=10, minRadius=0, maxRadius=0)

    center = center_coords
    radius = 14.1

    cx = center[0]
    cy = center[1]
    angle = np.linspace(0, 2 * np.pi, 180)
    X = (np.round(cx + radius * np.cos(angle))).astype(int)
    Y = (np.round(cy + radius * np.sin(angle))).astype(int)
    pts_reg = np.transpose(np.array([Y, X]))
    pts = [pts_reg[0]]
    curr_pt = pts_reg[0]
    for i in np.arange(1, len(pts_reg)):
        if pts_reg[i][0] == curr_pt[0] and pts_reg[i][1] == curr_pt[1]:
            continue
        else:
            pts.append(pts_reg[i])
            curr_pt = pts_reg[i]
    pts = np.array(pts)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].axis('off')
    ax[0].imshow(im2, cmap='gray', vmin=-600, vmax=2500)
    # ax[0].scatter(center[0], center[1], color='red', s=20)
    circ = plt.Circle(center, radius=radius, fill=False, edgecolor='red')
    ax[0].add_artist(circ)

    values = np.zeros(len(pts))
    for i in range(len(pts)):
        values[i] = im2[pts[i][0], pts[i][1]]

    # Find the distance between coordinates in pixel space, then transfer to mm
    dist = np.zeros(len(pts))
    for i in range(len(pts[:-1])):
        dist[i + 1] = dist[i] + np.sqrt((pts[i][0] - pts[i + 1][0]) ** 2 + (pts[i][1] - pts[i + 1][1]) ** 2)
    dist = dist * px_sz

    dist_int = np.linspace(0, dist[-1], 200)
    f = interp1d(dist, values, kind='linear')
    # f = CubicSpline(dist, values)
    # ax[2].scatter(dist, values)
    ax[1].plot(dist_int, f(dist_int))
    ax[1].set_xlabel('Distance (mm)')
    ax[1].set_ylabel('HU')
    ax[0].annotate('A', (7, 203), xycoords='axes points', color='white', fontsize=14)
    ax[1].annotate('B', (7, 203), xycoords='axes points', color='black', fontsize=14)

    fig.savefig(os.path.join(directory, folder, sub, 'fig', f'fig_{idx}{append}_14.1.png'), dpi=500)




