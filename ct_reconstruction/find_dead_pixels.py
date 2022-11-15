import os
import numpy as np
import matplotlib.pyplot as plt
import general_functions as gen


def find_dead_pixels(data1, data2, dark):
    """
    This function will take your attenuation data from a PCD acquisition and find the dead pixels and return the mask
    of the dead pixels (data must be corrected for air to obtain the attenuation data
    :param data: ndarray
            Must have at least 2 frames of data so that the mean pixel value can be found
            Just need the last bin (EC bin)
            Shape: <frames, 24, 576>
    :return: mask
            2D Mask with np.nan where there are dead pixels
    """

    dpm = np.abs(np.log(data1) - np.log(data2)) * 100

    data = np.add(data1, data2)
    med_data = np.nanmedian(data, axis=(0, 1))
    min_data = med_data / 2
    max_data = med_data * 2

    dpm[data > max_data] = np.nan
    dpm[data < min_data] = np.nan

    dpm[dpm >= 0.5] = np.nan
    dpm[dpm < 0.5] = 1

    dark[dark >= 50] = np.nan
    dark[dark < 50] = 1

    dpm = dpm * dark

    return dpm


def find_dead_pixels_all_bins(data, dark):
    """
    This function will take your counts data from a PCD acquisition and find the dead pixels and return the mask
    of the dead pixels
    :param data: ndarray
            You'll need to have an airscan with 2 frames (change the first two lines of the code because I use airscans
            with 12 frames)
            Shape: <frames, 24, 576, bins>
    :param dark: ndarray
            A dark scan with the same shape as data (without the frames)
            Shape: <24, 576, bins>
    :return: mask
            3D Mask with np.nan where there are dead pixels <rows, columns, bins>
    """

    data1 = np.sum(data[6:], axis=0)
    data2 = np.sum(data[0:6], axis=0)
    data = np.sum(data, axis=0)

    dpm = np.abs(np.log(data1) - np.log(data2)) * 100

    med_data = np.nanmedian(data, axis=(0, 1))
    min_data = med_data / 2
    max_data = med_data * 2

    for b in range(np.shape(data1)[-1]):
        dpm[:, :, b][data[:, :, b] > max_data[b]] = np.nan
        dpm[:, :, b][data[:, :, b] < min_data[b]] = np.nan

    dpm[dpm >= 0.5] = np.nan
    dpm[dpm < 0.5] = 1

    dark[dark >= 50] = np.nan
    dark[dark < 50] = 1

    dpm = dpm * dark

    dpm[:, :, -1] = np.prod(dpm[:, :, 0:-1], axis=-1)

    return dpm
