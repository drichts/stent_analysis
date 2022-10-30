import _pickle as pickle
import numpy as np
from scipy.io import savemat, loadmat
import os
import matplotlib.pyplot as plt
import mask_functions as msk
from matplotlib.patches import Rectangle


def crop_array(data):
    """
    This function will take a numpy array and crop it by allowing the user to click on the 4 corners of image
    :param data:
    :return:
    """

    # Click 4 corners of the points you want to crop the image to
    coords = msk.click_image(data, message_num=6)

    # Obtain the corners of the images from the 4 points (correcting for any skew)
    coords = np.squeeze(coords)
    x_max = int(round(np.max(coords[:, 1])))
    x_min = int(round(np.min(coords[:, 1])))

    y_max = int(round(np.max(coords[:, 0])))
    y_min = int(round(np.min(coords[:, 0])))

    # Crop the data
    cropped_data = data[x_min:x_max, y_min:y_max]

    # Plot to verify the cropped
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(data, cmap='gray')

    corner = (y_min - 0.5, x_min - 0.5)
    height = y_max - y_min + 1
    width = x_max - x_min + 1
    sq = Rectangle(corner, height, width, fill=False, edgecolor='red')
    ax[0].add_artist(sq)

    ax[1].imshow(cropped_data)

    plt.pause(5)
    plt.close()

    return np.array([x_min, x_max, y_min, y_max])


def intensity_correction(data, air_data, dark_data):
    """
    This function corrects flatfield data to show images, -ln(I/I0), I is the intensity of the data, I0 is the
    intensity in an airscan
    :param data: The data to correct (must be the same shape as air_data)
    :param air_data: The airscan data (must be the same shape as data)
    :param dark_data: The darkscan data (must be the same shape as airdata)
    :return: The corrected data array
    """
    with np.errstate(invalid='ignore'):
        data = np.log(np.subtract(air_data, dark_data)) - np.log(np.subtract(data, dark_data))
    return data


def cnr(image, contrast_mask, background_mask):
    """
    This function calculates the CNR of an ROI given the image, the ROI mask, and the background mask
    It also gives the CNR error
    :param image: The image to be analyzed as a 2D numpy array
    :param contrast_mask: The mask of the contrast area as a 2D numpy array
    :param background_mask: The mask of the background as a 2D numpy array
    :return CNR, CNR_error: The CNR and error of the contrast area
    """
    # The mean signal within the contrast area
    mean_roi = np.nanmean(image * contrast_mask)
    std_roi = np.nanstd(image * contrast_mask)

    # Mean and std. dev. of the background
    bg = np.multiply(image, background_mask)
    mean_bg = np.nanmean(bg)
    std_bg = np.nanstd(bg)

    cnr_val = abs(mean_roi - mean_bg) / std_bg
    cnr_err = np.sqrt(std_roi ** 2 + std_bg ** 2) / std_bg

    return cnr_val, cnr_err


def correct_dead_pixels(data, one_frame, one_bin, dead_pixel_mask):
    """
    This is to correct for known dead pixels. Takes the average of the eight surrounding pixels.
    Could implement a more sophisticated algorithm here if needed.
    :param data: 4D ndarray
                The data array in which to correct the pixels <captures, rows, columns, counter>,
                <captures, rows, columns> if there is only 1 set of data, or
                <rows, columns> for just one frame of one set of data
    :param one_frame: boolean
                True if there is only one bin, False if not
    :param one_bin: boolean
                True if there is only one bin, False if not
    :param dead_pixel_mask: 2D ndarray
                A data array with the same number of rows and columns as 'data'. Contains np.nan everywhere there
                is a known non-responsive pixel
    :return: The data array corrected for the dead pixels
    """

    if one_frame:
        data = np.expand_dims(data, axis=0)
    if one_bin:
        data = np.expand_dims(data, axis=3)
        dead_pixel_mask = np.expand_dims(dead_pixel_mask, axis=2)

    # Find the dead pixels (i.e pixels = to nan in the DEAD_PIXEL_MASK)
    dead_pixels = np.array(np.argwhere(np.isnan(dead_pixel_mask)), dtype='int')

    for p_idx, pixel in enumerate(dead_pixels):

        # Pixel is corrected in every counter and capture
        data[:, pixel[0], pixel[1], pixel[2]] = get_average_pixel_value(data, pixel, dead_pixel_mask)

    return np.squeeze(data)


def get_average_pixel_value(img, pixel, dead_pixel_mask):
    """
    Averages the dead pixel using the 8 nearest neighbours
    Checks the dead pixel mask to make sure each of the neighbors is not another dead pixel

    :param img: 4D array
                The projection image. Shape: <frames, rows, columns, bins>

    :param pixel: tuple (row, column)
                The problem pixel (is a 2-tuple)

    :param dead_pixel_mask: 2D numpy array
                Mask with 1 at good pixel coordinates and np.nan at bad pixel coordinates

    :return: the average value of the surrounding pixels
    """
    shape = np.shape(img)
    row, col, b = pixel

    vals = np.zeros((8, shape[0]))

    # Count the number of nans around the pixel, if above a certain number we'll include the 16 pixels surrounding the
    # immediate 8 pixels surrounding the pixel, it will be less if the pixel is on an edge
    num_nans = 0
    edge = False  # If the pixel is on edge or not (True if it is)

    # Grabs each of the neighboring pixel values and sets to nan if they are bad pixels or
    # outside the bounds of the image
    if col == shape[2] - 1:
        vals[0] = np.nan
        num_nans += 1
    else:
        vals[0] = img[:, row, col + 1, b] * dead_pixel_mask[row, col + 1, b]
        if np.isnan(dead_pixel_mask[row, col + 1, b]):
            num_nans += 1
    if col == 0:
        vals[1] = np.nan
        num_nans += 1
    else:
        vals[1] = img[:, row, col - 1, b] * dead_pixel_mask[row, col - 1, b]
        if np.isnan(dead_pixel_mask[row, col - 1, b]):
            num_nans += 1
    if row == shape[1] - 1:
        vals[2] = np.nan
        num_nans += 1
    else:
        vals[2] = img[:, row + 1, col, b] * dead_pixel_mask[row + 1, col, b]
        if np.isnan(dead_pixel_mask[row + 1, col, b]):
            num_nans += 1
    if row == 0:
        vals[3] = np.nan
        num_nans += 1
    else:
        vals[3] = img[:, row - 1, col, b] * dead_pixel_mask[row - 1, col, b]
        if np.isnan(dead_pixel_mask[row - 1, col, b]):
            num_nans += 1
    if col == shape[2] - 1 or row == shape[1] - 1:
        vals[4] = np.nan
        num_nans += 1
    else:
        vals[4] = img[:, row + 1, col + 1, b] * dead_pixel_mask[row + 1, col + 1, b]
        if np.isnan(dead_pixel_mask[row + 1, col + 1, b]):
            num_nans += 1
    if col == 0 or row == shape[1] - 1:
        vals[5] = np.nan
        num_nans += 1
    else:
        vals[5] = img[:, row + 1, col - 1, b] * dead_pixel_mask[row + 1, col - 1, b]
        if np.isnan(dead_pixel_mask[row + 1, col - 1, b]):
            num_nans += 1
    if col == shape[2] - 1 or row == 0:
        vals[6] = np.nan
        num_nans += 1
    else:
        vals[6] = img[:, row - 1, col + 1, b] * dead_pixel_mask[row - 1, col + 1, b]
        if np.isnan(dead_pixel_mask[row - 1, col + 1, b]):
            num_nans += 1
    if col == 0 or row == 0:
        vals[7] = np.nan
        num_nans += 1
    else:
        vals[7] = img[:, row - 1, col - 1, b] * dead_pixel_mask[row - 1, col - 1, b]
        if np.isnan(dead_pixel_mask[row - 1, col - 1, b]):
            num_nans += 1

    # If there are 4 or more values around the pixel that are nan, go 1 row out to calculate the average
    # if num_nans > 3:
    #     vals2 = secondary_pix_val(img, pixel, dead_pixel_mask)
    #     vals = np.concatenate((vals, vals2))

    # Takes the average of the neighboring pixels excluding nan values
    avg = np.nanmean(vals, axis=0)

    return avg


def secondary_pix_val(img, pixel, dead_pixel_mask):
    """
        Returns the 16 nearest neighbours, excluding the 8 immediately surrounding the pixel
        Checks the dead pixel mask to make sure each of the neighbors is not another dead pixel

        :param img: 4D array
                    The projection image. Shape: <frames, rows, columns, bins>

        :param pixel: tuple (row, column)
                    The problem pixel (is a 2-tuple)

        :param dead_pixel_mask: 2D numpy array
                    Mask with 1 at good pixel coordinates and np.nan at bad pixel coordinates

        :return: vals: 2D ndarray
                    The 16 pixel values
    """

    shape = np.shape(img)
    row, col, b = pixel

    # Array to hold the 16 values
    vals = np.zeros((16, shape[0]))

    if col > shape[2] - 3:  # If pixel is in the last two columns, but not in a corner
        vals[0] = np.nan
        vals[1] = np.nan
        vals[2] = np.nan
    else:
        if row == 0:
            vals[0] = np.nan
        else:
            vals[0] = img[:, row - 1, col + 2, b] * dead_pixel_mask[row - 1, col + 2, b]
        vals[1] = img[:, row, col + 2, b] * dead_pixel_mask[row, col + 2, b]
        if row == shape[1] - 1:
            vals[2] = np.nan
        else:
            vals[2] = img[:, row + 1, col + 2, b] * dead_pixel_mask[row + 1, col + 2, b]
    if row > shape[1] - 3:  # If the pixel is in the last two rows, but not in a corner
        vals[4] = np.nan
        vals[5] = np.nan
        vals[6] = np.nan
    else:
        if col == shape[2] - 1:
            vals[4] = np.nan
        else:
            vals[4] = img[:, row + 2, col + 1, b] * dead_pixel_mask[row + 2, col + 1, b]
        vals[5] = img[:, row + 2, col, b] * dead_pixel_mask[row + 2, col, b]
        if col == 0:
            vals[6] = np.nan
        else:
            vals[6] = img[:, row + 2, col - 1, b] * dead_pixel_mask[row + 2, col - 1, b]
    if col < 2:  # If the pixel is in the first two columns, but not in a corner
        vals[8] = np.nan
        vals[9] = np.nan
        vals[10] = np.nan
    else:
        if row == 0:
            vals[10] = np.nan
        else:
            vals[10] = img[:, row - 1, col - 2, b] * dead_pixel_mask[row - 1, col - 2, b]
        vals[9] = img[:, row, col - 2, b] * dead_pixel_mask[row, col - 2, b]
        if row == shape[1] - 1:
            vals[8] = np.nan
        else:
            vals[8] = img[:, row + 1, col - 2, b] * dead_pixel_mask[row + 1, col - 2, b]
    if row < 2:  # If the pixel is in the first two rows, but not in a corner
        vals[12] = np.nan
        vals[13] = np.nan
        vals[14] = np.nan
    else:
        if col == shape[2] - 1:
            vals[14] = np.nan
        else:
            vals[14] = img[:, row - 2, col + 1, b] * dead_pixel_mask[row - 2, col + 1, b]
        vals[13] = img[:, row - 2, col, b] * dead_pixel_mask[row - 2, col, b]
        if col == 0:
            vals[12] = np.nan
        else:
            vals[12] = img[:, row - 2, col - 1, b] * dead_pixel_mask[row - 2, col - 1, b]
    if col > shape[2] - 3 or row > shape[1] - 3:  # If the pixel is in the lower right corner
        vals[3] = np.nan
    else:
        vals[3] = img[:, row + 2, col + 2, b] * dead_pixel_mask[row + 2, col + 2, b]
    if col < 2 or row > shape[1] - 3:  # If the pixel is in the lower left corner
        vals[7] = np.nan
    else:
        vals[7] = img[:, row + 2, col - 2, b] * dead_pixel_mask[row + 2, col - 2, b]
    if col > shape[2] - 3 or row < 2:  # If the pixel is in the upper right corner
        vals[15] = np.nan
    else:
        vals[15] = img[:, row - 2, col + 2, b] * dead_pixel_mask[row - 2, col + 2, b]
    if col < 2 or row < 2:  # If the pixel is in the upper left corner
        vals[11] = np.nan
    else:
        vals[11] = img[:, row - 2, col - 2, b] * dead_pixel_mask[row - 2, col - 2, b]

    return vals

# data = np.random.random((7, 5, 5, 2))
# for i in range(5):
#     for j in range(5):
#         print(i, j)
#         print(data[0, :, :, 0])
#         vals = secondary_pix_val(data, (i, j, 0), np.ones((5, 5, 2)))
#         print(vals[:, 0])
#         print()


def correct_leftover_pixels(data, one_frame, one_bin):
    """
    This will correct for any other nan or inf pixels
    :param data: 4D ndarray
                The data array in which to correct the pixels <captures, rows, columns, counter>,
                <captures, rows, columns> if there is only 1 set of data, or
                <rows, columns> for just one frame of one set of data
    :param one_frame: boolean
                True if there is only one bin, False if not
    :param one_bin: boolean
                True if there is only one bin, False if not
    :param dpm: 2D ndarray
                A data array with the same number of rows and columns as 'data'. Contains np.nan everywhere there
                is a known non-responsive pixel
    :return: The data array corrected for the dead pixels
    """

    if one_frame:
        data = np.expand_dims(data, axis=0)
    if one_bin:
        data = np.expand_dims(data, axis=3)

    # This will find any left over nan values and correct them
    data[np.isinf(data)] = np.nan  # Set inf values to nan
    nan_coords = np.argwhere(np.isnan(data))
    num_nan = len(nan_coords)
    while num_nan > 0:
        print(f'Correcting secondary nan coords: {num_nan} left')
        for c_idx, coords in enumerate(nan_coords):
            coords = tuple(coords)
            frame = coords[0]
            pixel = coords[-3:-1]
            img_bin = coords[-1]
            temp_img = data[frame, :, :, img_bin]

            data[coords] = get_average_single_pixel_value(temp_img, pixel, np.ones((24, 576)))

        nan_coords = np.argwhere(np.isnan(data))
        print(f'After correction: {len(nan_coords)} dead pixels left')
        print()
        # If we can't correct for the remaining coordinates break the loop
        if len(nan_coords) == num_nan:
            print(f'Broke because the number of nan pixels remained the same: {num_nan}')
            break
        num_nan = len(nan_coords)

    return data


def get_average_single_pixel_value(img, pixel, dead_pixel_mask):
    """
    Averages the dead pixel using the 8 nearest neighbours
    Checks the dead pixel mask to make sure each of the neighbors is not another dead pixel
    :param img: 2D array
                The projection image
    :param pixel: tuple (row, column)
                The problem pixel (is a 2-tuple)
    :param dead_pixel_mask: 2D numpy array
                Mask with 1 at good pixel coordinates and np.nan at bad pixel coordinates
    :return: the average value of the surrounding pixels
    """
    shape = np.shape(img)
    row, col = pixel

    # Grabs each of the neighboring pixel values and sets to nan if they are bad pixels or
    # outside the bounds of the image
    if col == shape[1] - 1:
        n1 = np.nan
    else:
        n1 = img[row, col + 1] * dead_pixel_mask[row, col + 1]
    if col == 0:
        n2 = np.nan
    else:
        n2 = img[row, col - 1] * dead_pixel_mask[row, col - 1]
    if row == shape[0] - 1:
        n3 = np.nan
    else:
        n3 = img[row + 1, col] * dead_pixel_mask[row + 1, col]
    if row == 0:
        n4 = np.nan
    else:
        n4 = img[row - 1, col] * dead_pixel_mask[row - 1, col]
    if col == shape[1] - 1 or row == shape[0] - 1:
        n5 = np.nan
    else:
        n5 = img[row + 1, col + 1] * dead_pixel_mask[row + 1, col + 1]
    if col == 0 or row == shape[0] - 1:
        n6 = np.nan
    else:
        n6 = img[row + 1, col - 1] * dead_pixel_mask[row + 1, col - 1]
    if col == shape[1] - 1 or row == 0:
        n7 = np.nan
    else:
        n7 = img[row - 1, col + 1] * dead_pixel_mask[row - 1, col + 1]
    if col == 0 or row == 0:
        n8 = np.nan
    else:
        n8 = img[row - 1, col - 1] * dead_pixel_mask[row - 1, col - 1]

    # Takes the average of the neighboring pixels excluding nan values
    avg = np.nanmean(np.array([n1, n2, n3, n4, n5, n6, n7, n8]))

    return avg


def sumpxp(data, num_pixels):
    """
    This function takes a data array and sums nxn pixels along the row and column data
    :param data: 5D ndarray
                The full data array <captures, rows, columns, bins>
    :return: The new data array with nxn pixels from the inital data summed together
    """
    dat_shape = np.array(np.shape(data))
    dat_shape[-3] = int(dat_shape[-3] / num_pixels)  # Reduce size by num_pixels in the row and column directions
    dat_shape[-2] = int(dat_shape[-2] / num_pixels)

    ndata = np.zeros(dat_shape)
    n = num_pixels
    for row in np.arange(dat_shape[-3]):
        for col in np.arange(dat_shape[-2]):
            # Get each 2x2 subarray over all of the first 2 axes
            if len(dat_shape) == 4:
                temp = data[:, n * row:n * row + n, n * col:n * col + n, :]
                ndata[:, row, col, :] = np.nanmean(temp, axis=(-3, -2))  # Sum over only the rows and columns
            elif len(dat_shape) < 4:
                temp = data[n * row:n * row + n, n * col:n * col + n]
                ndata[row, col] = np.nanmean(temp, axis=(-3, -2))  # Sum over only the rows and columns
            else:
                print('Error: array size not found')
    return ndata


def reshape(data):
    new_shape = (data.shape[0] // 2, 2, *data.shape[1:])
    data_sum = np.sum(np.reshape(data, new_shape), axis=1)
    np.save(r'D:\OneDrive - University of Victoria\Research\LDA Data\ct_180frames_1sproj_111220 - Synth\Data\data.npy', data_sum)
    return data_sum


def save_mat(path, data):
    savemat(path, {'data': data, 'label': 'central-ish sinogram'})
