import os
import numpy as np
import scipy as sp
from scipy import ndimage
import matplotlib.pyplot as plt
import pydicom as pyd
from glob import glob
from natsort import natural_keys
import cv2
from find_nearest import find_nearest_index
from general_functions import crop_array
from mask_functions import click_image


def rotate_profile(center, degree, edge_pt):
    """

    :param center:
    :param degree:
    :param edge_pt:
    :return:
    """
    # Find the distance betweeen the edge and the center
    r = np.sqrt((center[0] - edge_pt[0])**2 + (center[1] - edge_pt[1])**2)

    num_profs = int(360 / degree)  # The number of profiles drawn from the center
    profiles = np.zeros((num_profs, 2))
    profiles[0] = edge_pt  # Set the first profile edge point to the given edge point

    # Find the rest of the edge points
    for idx, angle in enumerate(np.arange(degree, 360, degree)):
        rad = angle / 180 * np.pi
        profiles[idx+1] = [center[0] + np.round(r*np.sin(rad)), center[1] + np.round(r*np.cos(rad))]

    return profiles


def find_dist_with_fwhm(center, edge_pt, image, num_new_pts=100, vxl_sz=0.511):
    """

    :param center:
    :param edge_pt:
    :param image:
    :param num_new_pts:
    :param vxl_sz:
    :return:
    """

    x0, y0 = center
    x1, y1 = extend_point(center, edge_pt)

    xpts, ypts = np.linspace(x0, x1, num_new_pts), np.linspace(y0, y1, num_new_pts)

    # Extract the values along the line, using cubic interpolation
    intp_prof = ndimage.map_coordinates(image, np.vstack((ypts, xpts)))

    # Find the distance between the center and the edge point (edge_pt)
    dist = np.sqrt((y0 - y1)**2 + (x0 - x1)**2) * vxl_sz

    # Create an array from 0 to dist with the same number of points as the xpts and ypts
    dist_arr = np.linspace(0, dist, num_new_pts)

    # ax[1].plot(intp)
    # fig = plt.figure()
    fwhm_prof = intp_prof - np.min(intp_prof)
    fwhm_prof = fwhm_prof - np.max(fwhm_prof) / 2
    # plt.plot(dist_arr, intp_prof)

    idxmax = np.argmax(fwhm_prof)
    if idxmax == 0:
        radius, fwhm, outer_radius = -1, -1, -1
    else:
        idx1 = find_nearest_index(fwhm_prof[0:idxmax], 0)
        idx2 = find_nearest_index(fwhm_prof[idxmax:], 0) + idxmax
        # plt.scatter([dist_arr[idx1], dist_arr[idx2]], [intp_prof[idx1], intp_prof[idx2]])
        # # plt.pause(2)
        # # plt.close()

        # Find the distance from the center to the second FWHM point and between the two FWHM points
        radius = dist_arr[idx1]
        fwhm = dist_arr[idx2] - dist_arr[idx1]
        outer_radius = dist_arr[idx2]

    return radius, fwhm, outer_radius


def extend_point(center, edge_pt):
    """

    :param center:
    :param edge_pt:
    :return:
    """
    x1, y1 = center
    x2, y2 = edge_pt

    d = np.sqrt((center[0] - edge_pt[0])**2 + (center[1] - edge_pt[1])**2)

    m = (y2 - y1) / (x2 - x1)

    pt1 = np.zeros(2)
    pt2 = np.zeros(2)

    if x2 == x1:

        pt1[0] = x2
        pt2[0] = x2

        pt1[1] = y2 + d
        pt2[1] = y2 - d

    else:

        pt1[0] = x2 + (d / (np.sqrt(m**2 + 1)))
        pt2[0] = x2 - (d / (np.sqrt(m**2 + 1)))

        pt1[1] = y2 + m * (pt1[0] - x2)
        pt2[1] = y2 + m * (pt2[0] - x2)

    dist1 = np.sqrt((center[0] - pt1[0])**2 + (center[1] - pt1[1])**2)
    dist2 = np.sqrt((center[0] - pt2[0]) ** 2 + (center[1] - pt2[1]) ** 2)

    # print(dist, dist1, dist2)
    if dist1 > dist2:
        return pt1
    else:
        return pt2
