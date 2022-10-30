import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import interp1d, CubicSpline
from general_functions import crop_array
from mask_functions import click_image
import pydicom as pyd
from natsort import natural_keys
from glob import glob

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


# Go through the appropriate files and calculate the center, find an edge point and calculate width and diameter
for idx, val in enumerate(good_slices):
    file = files[val]

    data = pyd.dcmread(file)

    im = data.pixel_array * data.RescaleSlope + data.RescaleIntercept  # Get just the pixel array

    im1 = im[coords[0]:coords[1], coords[2]:coords[3]]
    im2 = np.copy(im1)
    im1 = ((im1 - im1.min()) * (1 / (im1.max() - im1.min()) * 255)).astype('uint8')
    im1 = cv2.medianBlur(im1, 5)
    circle = cv2.HoughCircles(im1, cv2.HOUGH_GRADIENT, 1, 20, param1=15, param2=10, minRadius=0, maxRadius=0)

    circle = circle[0]
    center = (circle[0, 0], circle[0, 1])
    radius = circle[0, 2] - 1

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

    fig.savefig(os.path.join(directory, folder, '10cm_phantom', sub, 'fig', f'profile_{idx}.png'), dpi=500)


