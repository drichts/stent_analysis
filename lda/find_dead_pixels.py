import os
import numpy as np
import matplotlib.pyplot as plt
import general_functions as gen

# This will find the dead pixels in the attenuation data (air corrected)


def blah(data):
    """
    This function will take your attenuation data from a PCD acquisition and find the dead pixels and return the mask
    of the dead pixels (data must be corrected for air to obtain the attenuation data
    :param data: ndarray
            Must have at least 2 frames of data so that the mean pixel value can be found
            Just need the last bin (EC bin)
            Shape: <frames, 24, 576>
    :return: mask
            Mask with np.nan where there are dead pixels
    """

    data_mean = np.mean(data, axis=0)  # Take the mean of the data over the frame axis

    # Compute the gradient over the rows and the columns
    grad_row = np.abs(np.gradient(data_mean, axis=0))
    grad_col = np.abs(np.gradient(data_mean, axis=1))

    # Set any pixels that have a gradient over 0.15 to be nan and any other to be 1
    grad_row[grad_row >= 0.2] = np.nan
    grad_col[grad_col >= 0.2] = np.nan
    grad_row[grad_row < 0.2] = 1
    grad_col[grad_col < 0.2] = 1

    # Compute the mask by multiplying the two gradients together
    mask = np.multiply(grad_row, grad_col)

    return mask, grad_row, grad_col


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

    # dpm[np.where(np.isfinite(dpm))] = 1
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

    # dpm[np.where(np.isfinite(dpm))] = 1
    dpm[dpm >= 0.5] = np.nan
    dpm[dpm < 0.5] = 1

    dark[dark >= 50] = np.nan
    dark[dark < 50] = 1

    dpm = dpm * dark

    dpm[:, :, -1] = np.prod(dpm[:, :, 0:-1], axis=-1)

    return dpm
