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


def ring_correction(data, air, dpm, num_bins=7):
    """
    This algorithm takes the projection image (data) of a uniform bottle of water and uses the horizontal profiles
    across each row of pixels to correct the same profiles in ct data projections
    :param data: ndarray
            The raw count data for the projection image of the bottle of water. <rows, columns, num_bins>
    :param air: ndarray
            The raw count data for the air scan: <rows, columns, num_bins>
    :param dpm: ndarray
            The dead pixel mask showing the location of the dead pixels, 1's where there are good pixels, np.nan
            where there are bad pixels <rows, columns, num_bins
    :param num_bins: int
            The number of energy bins the data contains
    :return: ndarray
            The array that will be multiplied to subsequent data to correct for slight pixel variations and correct for
            some ring artifacts
    """

    print('Correcting dead pixels in water scan')

    # Define a flag to give to the correction algorithms to show whether there is one energy bin or multiple
    if num_bins == 1:
        one_bin = True
    else:
        one_bin = False
    data = correct_dead_pixels(data, one_frame=True, one_bin=one_bin, dead_pixel_mask=dpm)

    # Correct the water projections for air
    corr = np.squeeze(np.log(air) - np.log(data))

    # Correct any leftover np.nan pixels
    corr = correct_leftover_pixels(np.squeeze(corr), one_frame=True, one_bin=one_bin)
    corr = np.squeeze(corr)

    # If there is only one bin, we need to expand the dimensions to have 1 bin in the 2nd index
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

    # The correction array for all other data is the ratio of the corrected over the old data
    corr_array = np.divide(corrected_array, corr)

    return np.squeeze(corr_array)

