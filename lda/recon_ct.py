import os
import numpy as np
import matplotlib.pyplot as plt
import tigre.algorithms as algs
from scipy.interpolate import interp1d
from lda.pcd_geom import PCDGeometry
from lda.xcat_geom import XCATGeometry
from lda.ring_corr import devon_correction
import general_functions as gen
from lda.convert_air_dark import convert
from lda.find_dead_pixels import find_dead_pixels, find_dead_pixels_all_bins
from attenuation.filter_spectrum import filter_spectrum
from skimage.restoration import denoise_tv_chambolle

# VARIABLES TO CHANGE
dir_folder = '22_10_11_CT_stents_heli'
scan_folder = 'pink_bottom'

# VARIABLES YOU MAY NEED TO CHANGE
# data_directory = r'C:\Users\drich\OneDrive - University of Victoria\Research\LDA Data'
data_directory = r'D:\OneDrive - University of Victoria\Research\LDA Data'
num_angles = 2880  # The number of projections
scan_duration = 360  # The length of the CT scan (s)
num_bins = 7  # The number of energy bins including the EC bin
offset = -1
thresholds = [35, 52, 67, 82, 95, 120]
roll = 380  # The number of angles to roll the sinogram
filter_type = 'Al'  # Al or Cu most likely
filter_thickness = 6
append = f''
recon_filter = 'shepp_logan'  # hann, hamming, cosine, shepp_logan
num_z_slices = 7
num_side_pxs = 423
save = True
kedge_bins = [1, 2]
kedge_mat = 'Ta'


def prepare_folders(folder=dir_folder, sub=scan_folder, directory=data_directory):
    # Check if folder actually exists before creating new subfolders within it
    if not os.path.exists(os.path.join(directory, folder, sub)):
        raise Exception(f'Folder does not exist: {os.path.join(directory, folder, sub)}')
    os.makedirs(os.path.join(directory, folder, sub, 'fig'), exist_ok=True)
    os.makedirs(os.path.join(directory, folder, sub, 'CT'), exist_ok=True)


def prepare_sinogram(airtime=60, duration=scan_duration, num_proj=num_angles, n_bins=num_bins, corr_rings=True,
                     folder=dir_folder, sub=scan_folder, directory=data_directory, append='', save=True):
    """
    This function will do the air correction, correct any dead pixels in the data, and apply the ring artifact
    correction if desired
    :param airtime:
    :param duration:
    :param num_proj:
    :param n_bins:
    :param corr_rings:
    :param folder:
    :param sub:
    :param directory:
    :return:
    """
    # Convert the 65 s air and dark scans into a single frame 60 s air and dark scan
    convert(folder, directory=directory)

    # Load the raw data
    data = np.load(os.path.join(directory, folder, sub, 'Data', 'data.npy'))
    print('Raw data loaded')

    # Load the air data and scale the counts by the appropriate time
    proj_time = duration/num_proj  # The scale factor (the scan time per projection/angle)
    air_data = np.load(os.path.join(directory, folder, 'airscan_60s', 'Data', 'data.npy')) / (airtime / proj_time)

    # Set the appropriate number of frames
    shp = np.shape(data)
    if n_bins == 1:
        one_bin = True
        if len(shp) == 3:
            one_frame = False
        else:
            one_frame = True
    else:
        one_bin = False
        # Get rid of the EC bin and replace the pileup bin with the summation of the counts of all other bins
        air_data = air_data[:, :, 0:6]
        air_data[:, :, 5] = np.sum(air_data[:, :, 0:5], axis=2)
        if len(shp) == 3:
            one_frame = True
            data = data[:, :, 0:6]
            data[:, :, 5] = np.sum(data[:, :, 0:5], axis=2)
        else:
            one_frame = False
            data = data[:, :, :, 0:6]
            data[:, :, :, 5] = np.sum(data[:, :, :, 0:5], axis=3)
        n_bins -= 1

    # Load or create the appropriate deadpixel mask
    if os.path.exists(os.path.join(directory, folder, 'dead_pixel_mask.npy')):
        dpm = np.load(os.path.join(directory, folder, 'dead_pixel_mask.npy'))
    else:
        dpm_air = np.load(os.path.join(directory, folder, 'airscan_65s', 'Data', 'data.npy'))[1:, :, :, 0:6]
        dpm_air[:, :, :, 5] = np.sum(dpm_air[:, :, :, 0:5], axis=3)
        dpm_dark = np.load(os.path.join(directory, folder, 'darkscan_60s', 'Data', 'data.npy'))[:, :, 0:6]
        dpm_dark[:, :, 5] = np.sum(dpm_dark[:, :, 0:5], axis=2)
        dpm = find_dead_pixels_all_bins(dpm_air, dpm_dark)
        if save:
            np.save(os.path.join(directory, folder, 'dead_pixel_mask.npy'), dpm)

        # Show the deadpixel mask
        for i in range(n_bins):
            fig_dpm = plt.figure()
            plt.imshow(dpm[:, :, i], interpolation='nearest')
            plt.title(f'Dead Pixel Mask Bin {i}')
            plt.pause(2)
            if save:
                fig_dpm.savefig(os.path.join(directory, folder, sub, 'fig', f'dead_pixel_mask_{i}.png'), dpi=500)
            plt.close()

    # This will cut the projections down to the correct number if there are more than necessary
    # if num_proj != len(data):
    #     diff = abs(num_proj - len(data))
    #     data = data[int(np.ceil(diff / 2)):len(data) - diff // 2]

    data = data[5:-35]

    # Do the -ln(I/I0) correction
    # Correct raw data and air data for dead pixels
    print('Correcting dead pixels in raw data')
    data = correct_dead_pixels(data, dpm, one_bin, one_frame)
    print('Correcting dead pixels in air scan')
    air_data = correct_dead_pixels(air_data, dpm, False, True)

    sino = np.log(air_data) - np.log(data)
    sino = gen.correct_leftover_pixels(sino, one_frame=one_frame, one_bin=one_bin)
    print('Data corrected for air')

    # Correct for pixel non-uniformities (ring artifacts)
    if corr_rings:
        print(f'Median bin values (before ring corr): {np.nanmedian(np.sum(sino, axis=0), axis=(0, 1))}')
        # Check if the correction exists, if not make it
        if os.path.exists(os.path.join(directory, folder, 'corr_array.npy')):
            sino = np.multiply(np.load(os.path.join(directory, folder, 'corr_array.npy')), sino)
        else:
            # Get rid of the EC bin and replace the pileup bin with the summed bins
            corr_data = np.load(os.path.join(directory, folder, 'water', 'Data', 'data.npy'))[:, :, :, 0:6]
            corr_data = np.sum(corr_data[1:], axis=0)
            corr_data[:, :, 5] = np.sum(corr_data[:, :, 0:5], axis=-1)

            corr_array = devon_correction(corr_data, air_data * (airtime / proj_time), dpm, num_bins=n_bins)
            if save:
                np.save(os.path.join(directory, folder, 'corr_array.npy'), corr_array)
            sino = np.multiply(corr_array, sino)

        print(f'Median bin values (after ring corr): {np.nanmedian(np.sum(sino, axis=0), axis=(0, 1))}')

    # Correct for the central pixels
    sino = final_sinogram_correction(sino, n_bins)
    sino = gen.correct_leftover_pixels(sino, one_frame=one_frame, one_bin=one_bin)

    # Save the sinogram and display it
    if save:
        np.save(os.path.join(directory, folder, sub, 'Data', f'sinogram{append}.npy'), sino)
    fig_sino = plt.figure()
    plt.imshow(np.roll(sino[:, 14, :, -1], 200, axis=0), interpolation='nearest')
    plt.title('Summed bin sinogram')
    plt.pause(5)
    if save:
        fig_sino.savefig(os.path.join(directory, folder, sub, 'fig', f'sinogram{append}.png'), dpi=500)
    plt.close()

    return sino


def prepare_eid_sinogram(thresholds, airtime=60, duration=scan_duration, num_proj=num_angles, corr_rings=True,
                     folder=dir_folder, sub=scan_folder, directory=data_directory, append='', save=True):
    """
    This function will do the air correction, correct any dead pixels in the data, and apply the ring artifact
    correction if desired
    :param airtime:
    :param duration:
    :param num_proj:
    :param corr_rings:
    :param folder:
    :param sub:
    :param directory:
    :return:
    """
    # Convert the 65 s air and dark scans into a single frame 60 s air and dark scan
    convert(folder, directory=directory)

    # Load the raw data
    data = np.load(os.path.join(directory, folder, sub, 'Data', 'data.npy'))
    print('Raw data loaded')

    # Load the air data and scale the counts by the appropriate time
    proj_time = duration/num_proj  # The scale factor (the scan time per projection/angle)
    air_data = np.load(os.path.join(directory, folder, 'airscan_60s', 'Data', 'data.npy')) / (airtime / proj_time)

    # Multiply the bins by the lower threshold and add together
    thresholds = thresholds[:-1]
    air_data = air_data[:, :, 0:5] * thresholds
    air_data = np.sum(air_data, axis=2)

    data = data[:, :, :, 0:5] * thresholds
    data = np.sum(data, axis=3)

    # Load or create the appropriate deadpixel mask
    if os.path.exists(os.path.join(directory, folder, 'dead_pixel_mask_eid.npy')):
        dpm = np.load(os.path.join(directory, folder, 'dead_pixel_mask_eid.npy'))
    else:
        dpm_air = np.load(os.path.join(directory, folder, 'airscan_65s', 'Data', 'data.npy'))[1:, :, :, 0:5]
        dpm_air = dpm_air * thresholds
        dpm_air1 = np.sum(dpm_air[6:], axis=(0, 3))
        dpm_air2 = np.sum(dpm_air[0:6], axis=(0, 3))
        dpm_dark = np.load(os.path.join(directory, folder, 'darkscan_60s', 'Data', 'data.npy'))[:, :, 0:5]
        dpm_dark = np.sum(dpm_dark, axis=2)
        dpm = find_dead_pixels(dpm_air1, dpm_air2, dpm_dark)
        if save:
            np.save(os.path.join(directory, folder, 'dead_pixel_mask_eid.npy'), dpm)

        # Show the deadpixel mask
        fig_dpm = plt.figure()
        plt.imshow(dpm, interpolation='nearest')
        plt.title('Dead Pixel Mask EID')
        plt.pause(2)
        if save:
            fig_dpm.savefig(os.path.join(directory, folder, sub, 'fig', 'dead_pixel_mask_eid.png'), dpi=500)
        plt.close()

    # This will cut the projections down to the correct number if there are more than necessary
    if num_proj != len(data):
        diff = abs(num_proj - len(data))
        data = data[int(np.ceil(diff / 2)):len(data) - diff // 2]

    # Do the -ln(I/I0) correction
    # Correct raw data and air data for dead pixels
    print('Correcting dead pixels in raw data')
    data = correct_dead_pixels(data, dpm, True, False)
    print('Correcting dead pixels in air scan')
    air_data = correct_dead_pixels(air_data, dpm, True, True)

    sino = np.log(air_data) - np.log(data)
    sino = np.squeeze(gen.correct_leftover_pixels(sino, one_frame=False, one_bin=False))
    print('Data corrected for air')

    # Correct for pixel non-uniformities (ring artifacts)
    if corr_rings:
        print(f'Median bin values (before ring corr): {np.nanmedian(np.sum(sino, axis=0), axis=(0, 1))}')
        # Check if the correction exists, if not make it
        if os.path.exists(os.path.join(directory, folder, 'corr_array_eid.npy')):
            sino = np.multiply(np.load(os.path.join(directory, folder, 'corr_array_eid.npy')), sino)
        else:
            corr_data = np.load(os.path.join(directory, folder, 'water', 'Data', 'data.npy'))[1:, :, :, 0:5]
            corr_data = corr_data * thresholds
            corr_data = np.sum(corr_data, axis=(0, 3))

            corr_array = devon_correction(corr_data, np.squeeze(air_data) * (airtime / proj_time), dpm, num_bins=1)
            if save:
                np.save(os.path.join(directory, folder, 'corr_array_eid.npy'), corr_array)
            sino = np.multiply(corr_array, sino)

        print(f'Median bin values (after ring corr): {np.nanmedian(np.sum(sino, axis=0), axis=(0, 1))}')


    # Check for any last ring artifacts and correct for them
    sino = final_sinogram_correction(sino, 1, num_proj)
    sino = gen.correct_leftover_pixels(sino, one_frame=False, one_bin=True)

    # Save the sinogram and display it
    if save:
        np.save(os.path.join(directory, folder, sub, 'Data', f'sinogram{append}_eid.npy'), sino)
    fig_sino = plt.figure()
    plt.imshow(np.roll(sino[:, 14, :, -1], 200, axis=0), interpolation='nearest')
    plt.title('Summed bin sinogram')
    plt.pause(5)
    if save:
        fig_sino.savefig(os.path.join(directory, folder, sub, 'fig', f'sinogram{append}_eid.png'), dpi=500)
    plt.close()

    return sino


def prepare_TC_sinogram(airtime=60, duration=scan_duration, num_proj=num_angles, corr_rings=True,
                        folder=dir_folder, sub=scan_folder, directory=data_directory, append='', save=True):
    """
    This function will do the air correction, correct any dead pixels in the data, and apply the ring artifact
    correction if desired
    :param airtime:
    :param duration:
    :param num_proj:
    :param corr_rings:
    :param folder:
    :param sub:
    :param directory:
    :return:
    """
    # Convert the 65 s air and dark scans into a single frame 60 s air and dark scan
    convert(folder, directory=directory)

    # Load the raw data
    data = np.load(os.path.join(directory, folder, sub, 'Data', 'data.npy'))
    print('Raw data loaded')

    # Load the air data and scale the counts by the appropriate time
    proj_time = duration/num_proj  # The scale factor (the scan time per projection/angle)

    air_data = np.load(os.path.join(directory, folder, 'airscan_60s', 'Data', 'data.npy')) / (airtime / proj_time)

    # Acquire only the TC bin
    data = data[:, :, :, -1]
    air_data = air_data[:, :, -1]

    # Load or create the appropriate deadpixel mask
    if os.path.exists(os.path.join(directory, folder, 'dead_pixel_mask_TC.npy')):
        dpm = np.load(os.path.join(directory, folder, 'dead_pixel_mask_TC.npy'))
    else:
        dpm_air = np.load(os.path.join(directory, folder, 'airscan_65s', 'Data', 'data.npy'))[1:, :, :, -1]
        dpm_air1 = np.sum(dpm_air[6:], axis=0)
        dpm_air2 = np.sum(dpm_air[0:6], axis=0)
        dpm_dark = np.load(os.path.join(directory, folder, 'darkscan_60s', 'Data', 'data.npy'))[:, :, -1]
        dpm = find_dead_pixels(dpm_air1, dpm_air2, dpm_dark)
        if save:
            np.save(os.path.join(directory, folder, 'dead_pixel_mask_TC.npy'), dpm)

        # Show the deadpixel mask
        fig_dpm = plt.figure()
        plt.imshow(dpm, interpolation='nearest')
        plt.title('Dead Pixel Mask TC')
        plt.pause(2)
        if save:
            fig_dpm.savefig(os.path.join(directory, folder, sub, 'fig', 'dead_pixel_mask_TC.png'), dpi=500)
        plt.close()

    # This will cut the projections down to the correct number if there are more than necessary
    if num_proj != len(data):
        diff = abs(num_proj - len(data))
        data = data[int(np.ceil(diff / 2)):len(data) - diff // 2]

    # Do the -ln(I/I0) correction
    # Correct raw data and air data for dead pixels
    print('Correcting dead pixels in raw data')
    data = correct_dead_pixels(data, dpm, True, False)
    print('Correcting dead pixels in air scan')
    air_data = correct_dead_pixels(air_data, dpm, True, True)

    sino = np.log(air_data) - np.log(data)
    sino = np.squeeze(gen.correct_leftover_pixels(sino, one_frame=False, one_bin=False))
    print('Data corrected for air')

    # Correct for pixel non-uniformities (ring artifacts)
    if corr_rings:
        print(f'Median bin values (before ring corr): {np.nanmedian(np.sum(sino, axis=0), axis=(0, 1))}')
        # Check if the correction exists, if not make it
        if os.path.exists(os.path.join(directory, folder, 'corr_array_TC.npy')):
            sino = np.multiply(np.load(os.path.join(directory, folder, 'corr_array_TC.npy')), sino)
        else:
            corr_data = np.load(os.path.join(directory, folder, 'water', 'Data', 'data.npy'))[1:, :, :, -1]
            corr_data = np.sum(corr_data, axis=0)

            corr_array = devon_correction(corr_data, np.squeeze(air_data) * (airtime / proj_time), dpm, num_bins=1)
            if save:
                np.save(os.path.join(directory, folder, 'corr_array_TC.npy'), corr_array)
            sino = np.multiply(corr_array, sino)

        print(f'Median bin values (after ring corr): {np.nanmedian(np.sum(sino, axis=0), axis=(0, 1))}')


    # Check for any last ring artifacts and correct for them
    sino = final_sinogram_correction(sino, 1)
    sino = gen.correct_leftover_pixels(sino, one_frame=False, one_bin=True)

    # Save the sinogram and display it
    if save:
        np.save(os.path.join(directory, folder, sub, 'Data', f'sinogram{append}_TC.npy'), sino)
    fig_sino = plt.figure()
    plt.imshow(sino[:, 12, :, -1], interpolation='nearest')
    plt.title('TC bin sinogram')
    plt.pause(5)
    if save:
        fig_sino.savefig(os.path.join(directory, folder, sub, 'fig', f'sinogram{append}_TC.png'), dpi=500)
    plt.close()

    return sino


def correct_dead_pixels(data, dpm, one_bin, one_frame):
    """
    This is to correct for known dead pixels. Takes the average of the eight surrounding pixels.
    Could implement a more sophisticated algorithm here if needed.
    :return: The data array corrected for the dead pixels
    """

    data = gen.correct_dead_pixels(data, one_frame=one_frame, one_bin=one_bin, dead_pixel_mask=dpm)

    data = gen.correct_leftover_pixels(data, one_frame=one_frame, one_bin=one_bin)

    return data


def final_sinogram_correction(sinogram, num_bins):

    if num_bins == 1:
        sinogram = np.expand_dims(sinogram, axis=3)

    # Set the central 3 columns to nan so we can interpolate between them
    # sinogram[:, :, 286] = np.nan
    # sinogram[:, :, 287] = np.nan
    # sinogram[:, :, 288] = np.nan

    # See which pixels are bad
    for i in range(24):
        curr = np.sum(sinogram[:, i, :, -1], axis=0)
        g = np.gradient(curr)
        g = np.gradient(g)
        cols = np.squeeze(np.argwhere(g < -12))

        for c in cols:
            if 50 < c < 520:
                sinogram[:, i, c] = np.nan

    # # Hard coded pixels
    sinogram[:, 12, 411] = np.nan
    sinogram[:, 15, 365] = np.nan
    sinogram[:, 19, 541] = np.nan
    sinogram[:, 9, 75] = np.nan
    sinogram[:, 20, 226] = np.nan
    sinogram[:, 6, 519] = np.nan
    sinogram[:, 15, 431] = np.nan
    sinogram[:, 9, 343] = np.nan
    sinogram[:, 9, 344] = np.nan
    sinogram[:, 8, 343] = np.nan

    sinogram[:, 20, 326] = np.nan
    sinogram[:, 18, 382] = np.nan
    sinogram[:, 18, 324] = np.nan
    sinogram[:, 17, 328] = np.nan
    sinogram[:, 17, 379] = np.nan
    sinogram[:, 17, 363] = np.nan
    sinogram[:, 14, 259] = np.nan
    sinogram[:, 13, 346] = np.nan
    sinogram[:, 13, 369] = np.nan
    sinogram[:, 13, 251] = np.nan
    sinogram[:, 12, 212] = np.nan
    sinogram[:, 11, 300] = np.nan
    sinogram[:, 9, 363] = np.nan
    sinogram[:, 9, 369] = np.nan
    sinogram[:, 9, 380] = np.nan
    sinogram[:, 8, 365] = np.nan


    angs, rows, cols, bins = np.shape(sinogram)

    # Interpolate the traces in the sinogram
    for b in range(bins):
        for a in range(angs):
            for r in range(rows):

                # Find the column numbers that are not nans
                xpts = np.squeeze(np.argwhere(np.isfinite(sinogram[a, r, :, b])))
                # Find the column values for the column numbers that aren't nans
                ypts = sinogram[a, r, xpts, b]

                # Interpolate
                f = interp1d(xpts, ypts, kind='linear')
                xnew = np.squeeze(np.argwhere(np.isnan(sinogram[a, r])))

                sinogram[a, r, xnew, b] = f(xnew)

    return np.squeeze(sinogram)


def reconstruct_CT(sinogram, algorithm, n_bins=num_bins-1, filt='hamming', iterations=50,
                   h_offset=0, v_offset=0, dsd=578.0, dso=322.0, side=512, z_stack=24,
                   folder=dir_folder, sub=scan_folder, directory=data_directory, append='', save=True):
    """
    :param sinogram:
    :param algorithm: str
            Algorithm options: sart, sirt, ossart, ossart_tv, iterativereconalg, FDK, asd_pocs, awasd_pocs, cgls,
                               fista, ista, mlem
    :param n_bins: int
            The number of PCD bins in the data (including the total counts if using)
    :param filt: str, optional
            If using FDK, you may specify the filter type. Options: shep_logan, cosine, hamming, hann
    :param iterations: int, optional
            If using an iterative method, specify how many iterations to use. Defaults to 50
    :param h_offset: float
    :param v_offset:
    :param dsd:
    :param dso:
    :param folder:
    :param sub:
    :param directory:
    :return:
    """

    # Create the reconstruction geometry
    geo = PCDGeometry(DSD=dsd, DSO=dso, side=side, h_offset=h_offset, v_offset=v_offset, z_stack=z_stack)
    # geo = XCATGeometry()

    # Set the angles for reconstruction
    num_proj = len(sinogram)
    angles = np.linspace(0, 2 * np.pi, num_proj, endpoint=False)

    # Convert the sinogram to float32 if necessary
    sinogram = np.float32(sinogram)

    # Reconstruct
    ct = np.zeros((n_bins, z_stack, side, side))
    # ct = np.zeros((n_bins, 10, 1024, 1024))
    print(f'Starting reconstruction using {algorithm}')
    for b in range(n_bins):
        print(f'Reconstructing Bin {b}')
        if algorithm == 'FDK':
            if filter:
                ct[b] = algs.fdk(sinogram[:, :, :, b], geo, angles, filter=filt)
            else:
                ct[b] = algs.fdk(sinogram[:, :, :, b], geo, angles)
        else:
            ct[b] = eval(f'algs.{algorithm}(sinogram[:, :, :, b], geo, angles, iterations)')

        # Show the 12th slice to make sure the reconstruction is going well
        fig_ct = plt.figure(figsize=(6, 6))
        plt.imshow(ct[b, int(np.shape(ct)[0]/2)], cmap='gray', vmin=0.005, vmax=0.03, interpolation='nearest')
        plt.title(f'Raw Recon Bin {b}')
        if save:
            fig_ct.savefig(os.path.join(directory, folder, sub, 'fig', f'CT_Bin{b}{append}.png'), dpi=500)
        plt.pause(2)
        plt.close()

    # Save the CT data
    if save:
        np.save(os.path.join(directory, folder, sub, 'CT', f'CT_{algorithm}{append}.npy'), ct)

    return ct


def reconstruct_kedge(sinogram, kedge_bins, energies=thresholds, algorithm='FDK', material='Au', filter='hamming',
                      iterations=50, h_offset=0, v_offset=0, dsd=578.0, dso=322.0, side=512, z_stack=24,
                      folder=dir_folder, sub=scan_folder, directory=data_directory, save=True, append=''):
    """
    This is implenting the K-edge decomposition method from the Zhang et al paper from 2020
    :param sinogram:
    :param kedge_bins: list
            The two bins that have a threshold on the K-edge energy. Must use python indexing, eg [2, 3] for bins 3 and
            4
    :param energies: list
            The six energy threshold values in order (in keV)
    :param algorithm: str
            Algorithm options: sart, sirt, ossart, ossart_tv, iterativereconalg, FDK, asd_pocs, awasd_pocs, cgls,
                               fista, ista, mlem
    :param filter: str, optional
            If using FDK, you may specify the filter type. Options: shep_logan, cosine, hamming, hann
    :param iterations: int, optional
            If using an iterative method, specify how many iterations to use. Defaults to 50
    :param h_offset: float
    :param v_offset:
    :param dsd:
    :param dso:
    :param folder:
    :param sub:
    :param directory:
    :return:
    """

    # Fetch the material and water mass attenuation coefficients
    # att_folder = r'C:\Users\drich\OneDrive - University of Victoria\Research\Attenuation Data\K-edge Decomposition'
    att_folder = r'D:\OneDrive - University of Victoria\Research\Attenuation Data\K-edge Decomposition'
    mat_att = np.loadtxt(os.path.join(att_folder, 'K-edge materials', f'{material}.txt'))
    water_att = np.loadtxt(os.path.join(att_folder, 'Background materials', 'H2O.txt'))

    # Fetch the spectra for the weights
    # spectrum = np.load(r'C:\Users\drich\OneDrive - University of Victoria\Research\Material Decomposition\Beam Spectrum\mat_decomp_spectra_120kVp.npy')
    spectrum = np.load(r'D:\OneDrive - University of Victoria\Research\Material Decomposition\Beam Spectrum\mat_decomp_spectra_120kVp.npy')
    spectrum[:, 1] = spectrum[:, 1] * 1E10  # Scale the weights up

    # Filter the spectra
    spectrum = filter_spectrum(spectrum, filter_type, filter_thickness)
    spectrum = spectrum[:, 1]

    # Translate from MeV to keV for the energies
    mat_att[:, 0] = mat_att[:, 0] * 1000
    water_att[:, 0] = water_att[:, 0] * 1000

    # Look for the closest energy to the 3 thresholds
    idx = []

    for ei, energy in enumerate(energies[kedge_bins[0]:kedge_bins[1] + 2]):
        idx.append(np.argmin(np.abs(water_att[:, 0] - energy)))

    low_mat = np.mean(mat_att[idx[0]:idx[1] + 1, 1])
    high_mat = np.mean(mat_att[idx[1]:idx[2] + 1, 1])
    low_water = np.mean(water_att[idx[0]:idx[1] + 1, 1])
    high_water = np.mean(water_att[idx[1]:idx[2] + 1, 1])

    # Convert the sinogram to float32 if necessary
    sinogram = np.float32(sinogram)

    # K-edge decomposition
    high_bin = sinogram[:, :, :, kedge_bins[1]] * low_water
    low_bin = sinogram[:, :, :, kedge_bins[0]] * high_water

    sinogram = (high_bin - low_bin) / ((high_mat * low_water) - (low_mat * high_water))

    # Create the reconstruction geometry
    geo = PCDGeometry(DSD=dsd, DSO=dso, h_offset=h_offset, v_offset=v_offset, side=side, z_stack=z_stack)

    # Set the angles for reconstruction
    num_proj = len(sinogram)
    angles = np.linspace(0, 2 * np.pi, num_proj, endpoint=False)

    fig_sino = plt.figure()
    plt.imshow(sinogram[:, 12], interpolation='nearest')
    plt.title('KDA sinogram')
    plt.pause(5)
    if save:
        fig_sino.savefig(os.path.join(directory, folder, sub, 'fig', f'KDA_{material}_sinogram{append}.png'), dpi=500)
    plt.close()

    # Reconstruct
    print(f'Starting reconstruction using {algorithm}')

    if algorithm == 'FDK':
        if filter:
            ct = algs.fdk(sinogram, geo, angles, filter=filter)
        else:
            ct = algs.fdk(sinogram, geo, angles)
    else:
        ct = eval(f'algs.{algorithm}(sinogram, geo, angles, iterations)')

    # Show the 12th slice to make sure the reconstruction is going well
    fig_ct = plt.figure()
    plt.imshow(ct[int(np.shape(ct)[0]/2)], cmap='gray', interpolation='nearest')
    plt.title(f'Raw Recon K-edge {material}')

    if save:
        fig_ct.savefig(os.path.join(directory, folder, sub, 'fig', f'KDA_{material}{append}.png'), dpi=500)
    plt.pause(2)
    plt.close()

    # Save the CT data
    if save:
        np.save(os.path.join(directory, folder, sub, 'CT', f'KDA_{material}_FDK{append}.npy'), ct)

    return ct


def subtract_kedge(ct, kedge_bins, alg='FDK', material='Au', folder=dir_folder, sub=scan_folder, directory=data_directory):
    """

    :param ct:
    :param kedge_bins:
    :param material:
    :param folder:
    :param sub:
    :param directory:
    :return:
    """

    k = np.subtract(ct[kedge_bins[1]], ct[kedge_bins[0]])

    # Show the 12th slice to make sure the reconstruction is going well
    fig_ct = plt.figure()
    plt.imshow(k[4], cmap='gray', interpolation='nearest')
    plt.title(f'Raw Recon K-edge {material}')
    fig_ct.savefig(os.path.join(directory, folder, sub, 'fig', f'K-edge_{material}.png'), dpi=500)
    plt.pause(2)
    plt.close()

    # Save the CT data
    np.save(os.path.join(directory, folder, sub, 'CT', f'K-edge_{material}_{alg}.npy'), k)

    return k


if __name__ == '__main__':

    # for i in np.arange(15, 16):
    prepare_folders()
        # sf = f'rat_stent_{i}'
    # Normal recon
    sino = prepare_sinogram(save=save, append=append)
    # sino = np.load(os.path.join(data_directory, dir_folder, scan_folder, 'Data', 'sinogram.npy'))
    #     sino = np.load(os.path.join(data_directory, dir_folder, sf, 'Data', 'sinogram.npy'))
    # sino = np.roll(sino, roll, axis=0)

    # sino = np.expand_dims(sino, axis=3)

    # ct = reconstruct_CT(sino, algorithm='FDK', h_offset=offset, save=save, append=append, side=num_side_pxs,
    #                     z_stack=num_z_slices, filt=recon_filter)
    #     ct = reconstruct_CT(sino, algorithm='FDK', h_offset=offset, save=save, append=append, sub=sf)
    # ct = reconstruct_CT(sino, algorithm='FDK', n_bins=1, save=False)
    # for n in [718, 719, 721, 722]:
    #     sino = prepare_sinogram(save=save, append=f'{n}', num_proj=n)
    #     # sino = np.load(os.path.join(data_directory, dir_folder, scan_folder, 'Data', 'sinogram.npy'))
    #
    #     ct = reconstruct_CT(sino, algorithm='FDK', h_offset=offset, save=save, append=f'{n}')

    # K-edge recon
    # ct = np.load(os.path.join(data_directory, dir_folder, scan_folder, 'CT', f'CT_FDK{append}.npy'))
    # k = subtract_kedge(ct, kedge_bins, material=kedge_mat)
    # kda = reconstruct_kedge(sino, kedge_bins=kedge_bins, h_offset=offset, material=kedge_mat, side=num_side_pxs,
    #                         z_stack=num_z_slices, append=append, save=save, filter=recon_filter)

    # EID recon
    # sino = prepare_eid_sinogram(thresholds)
    # ct = reconstruct_CT(sino, algorithm='FDK', n_bins=1, h_offset=offset, append='_eid')
    #
    # TC bin recon
    # sino = prepare_TC_sinogram()
    # ct = reconstruct_CT(sino, algorithm='FDK', n_bins=1, h_offset=offset, append='_TC')
