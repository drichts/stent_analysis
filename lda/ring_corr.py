import numpy as np
from scipy import signal
from scipy.signal import medfilt
import matplotlib.pyplot as plt
from general_functions import correct_dead_pixels, correct_leftover_pixels, sumpxp


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.
    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal
    example:
    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    see also:
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    return y


def pixel_corr(data, num_bins=7, top=False, window='blackman'):
    """
    Jericho's correction method for correcting the variable pixel response of the detector
    :param data: ndarray
            Shape: (angles, rows, columns, bins) or (angles, rows, columns) The data the correction matrix is calculated
            rom (should be of auniform object)
    :param num_bins: int, optional
            The number of bins in the data array. Defaults to 7
    :param window: str, optional
            The type of window. Types: 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
    :return: ndarray
            The correction matrix to multiply the data by. Shape: (rows, columns, bins)
    """
    # Go through each of the bins

    if num_bins == 1:
        data = np.expand_dims(data, axis=3)

    full_corr_array = np.zeros(np.shape(data)[1:])

    for bin_num in np.arange(num_bins):

        # Crop away the top of the image with only air if imaging the top of the phantom
        if top:
            good_data = np.nanmean(data[:, 9:, :, bin_num], axis=0)
        else:
            good_data = np.nanmean(data[:, :, :, bin_num], axis=0)

        outliers = []

        # Find the mean of the data along the angle direction
        ref = np.nanmean(good_data, axis=0)

        # Now I'll discard the three biggest outliers and take that average.
        mins = np.argmax(np.abs(good_data - ref), axis=0)
        nn = 14

        for jj in range(nn):
            ref = np.nanmean(good_data, axis=0)
            mins = np.argmax(np.abs(good_data - ref), axis=0)
            outliers.append(mins)
            for ii in range(len(mins)):
                good_data[mins[ii], ii] = ref[ii]

        for jj in range(nn):
            for ii in range(len(mins)):
                good_data[outliers[jj][ii], ii] = np.nan

        # Mask invalid data, i.e. inf, nan, -inf, etc when taking the mean
        good_data = np.ma.masked_invalid(good_data)
        real_refs = np.mean(good_data, axis=0)

        smoothed = smooth(real_refs, window_len=10, window=window)

        smoothed3 = smoothed[4:-5]

        w = 0.1  # Cut-off frequency of the filter
        b, a = signal.butter(5, w, 'low')   # Numerator (b) and denominator (a) for Butterworth filter
        output = signal.filtfilt(b, a, real_refs)  # Apply the filter to the data

        smoothed3[25:-25] = output[25:-25]  # Replace filtered data in the
        if top:
            correction_array = np.nanmean(data[:, 10:, :, bin_num], axis=0) / smoothed3
            new_data = (data[:, 10:, :, bin_num] / correction_array).transpose(1, 2, 0)
        else:
            correction_array = np.nanmean(data[:, :, :, bin_num], axis=0)/smoothed3
            new_data = (data[:, :, :, bin_num] / correction_array).transpose(1, 2, 0)

        full_corr_array[:, :, bin_num] = correction_array

        new_data[new_data < -0.5] = 0
        image = new_data.copy()

        float_array = np.float32(10 * image.transpose(2, 0, 1))

        if top:
            data[:, 10:, :, bin_num] = float_array
        else:
            data[:, :, :, bin_num] = float_array

    return data, full_corr_array


def devon_correction(data, air, dpm, num_bins=7):

    # Correct the water projections for air
    print('Correcting dead pixels in water scan')
    if num_bins == 1:
        one_bin = True
    else:
        one_bin = False
    data = correct_dead_pixels(data, one_frame=True, one_bin=one_bin, dead_pixel_mask=dpm)
    # data = correct_leftover_pixels(np.squeeze(data), one_frame=True, one_bin=one_bin)

    # TEMPORARY
    # data = sumpxp(data, 6)

    corr = np.squeeze(np.log(air) - np.log(data))

    corr = correct_leftover_pixels(np.squeeze(corr), one_frame=True, one_bin=one_bin)
    corr = np.squeeze(corr)
    # TEMPORARY
    # print('TEMPORARY CHANGE TO RING CORR')
    # num_cut = 16
    # corr = corr[:, num_cut:-num_cut, :]
    # corr = corr[:, 45:-45, :]
    if one_bin:
        corr = np.expand_dims(corr, axis=2)

    num_rows = np.shape(corr)[0]
    num_cols = np.shape(corr)[1]

    # Create a smoothed signal
    corrected_array = np.zeros((num_rows, num_cols, num_bins))

    xpts = np.arange(num_cols)
    for row in range(num_rows):
        for bb in range(num_bins):
            curr_row = np.copy(corr[row, :, bb])
            med_row = medfilt(corr[row, :, bb], 21)  # Median filter of the current row

            # Go through all the values. If the actual data point significantly deviates from the median filter replace
            # it in the curr_row array
            for i in range(num_cols):
                if np.abs(corr[row, i, bb] - med_row[i]) > 0.15:
                    curr_row[i] = med_row[i]

            # Median filter the row corrected for significantly deviated points
            med_curr_row = medfilt(curr_row, 21)

            # Fit the median filtered corrected row and set the corrected array to the fit
            p = np.polyfit(xpts, med_curr_row, 6)

            corrected_array[row, :, bb] = np.polyval(p, xpts)
            # plt.plot(corrected_array[row, :, bb])
            # plt.plot(curr_row)
            # plt.pause(3)

    # The correction array for all other data is the ratio of the corrected over the old data
    corr_array = np.divide(corrected_array, corr)

    # ALSO TEMPORARY
    # ones = np.ones((24, 45, num_bins))
    # ones = np.ones((8, num_cut, num_bins))
    # corr_array = np.concatenate((ones, corr_array, ones), axis=1)

    return np.squeeze(corr_array)

