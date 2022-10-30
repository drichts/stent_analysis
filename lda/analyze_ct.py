import os
import numpy as np
import matplotlib.pyplot as plt
import mask_functions as msk
from lda.create_other_masks import create_masks
# from tomopy.misc.corr import remove_ring

# VARIABLES TO CHANGE
dir_folder = '22_10_11_CT_stents_heli'
scan_folder = 'pink_bottom'
save_note = ''
save = True
use_norm_vals = False  # If true, it will load the normalizing value arrays
save_norm_vals = False  # If true, it will same the water and air values in the un-normalized images as the norm values
kedge_mat = 'Ta'
slice_by_slice_norm = True

# VARIABLES YOU MAY NEED TO CHANGE
data_directory = r'D:\OneDrive - University of Victoria\Research\LDA Data'
# data_directory = r'C:\Users\drich\OneDrive - University of Victoria\Research\LDA Data'


def normalize_ct(ct, algorithm, water_slice=2, water_value=None, air_value=None, contrast_mask_type=None,
                 num_contrast_rois=None, folder=dir_folder, sub=scan_folder, directory=data_directory, append='',
                 save=True, save_norm=False, slice_by_slice=False):
    """

    :param ct:
    :param algorithm:
    :param water_slice:
    :param water_value:
    :param air_value:
    :param contrast_mask_type:
    :param num_contrast_rois:
    :param folder:
    :param sub:
    :param directory:
    :return:
    """
    # Create the Normalized CT folder
    # Check if folder actually exists before creating new subfolders within it
    if not os.path.exists(os.path.join(directory, folder, sub)):
        raise Exception(f'Folder does not exist: {os.path.join(directory, folder, sub)}')
    os.makedirs(os.path.join(directory, folder, sub, 'Norm CT'), exist_ok=True)

    data_shape = np.shape(ct)
    n_bins = data_shape[0]
    num_slices = data_shape[1]

    # Find the water vals within the phantom if they are not given in the function
    if water_value is None:
        # Get the mask for the water vials
        water_path = os.path.join(directory, folder, sub, f'water_mask{append}.npy')
        if os.path.exists(water_path):
            water_mask = np.load(water_path)
        else:
            water_mask = create_masks(ct[-1, water_slice], mask_type='water', folder=folder, sub=sub,
                                      directory=directory, save=save)

        # Find the water values for each of the bins
        if slice_by_slice:

            water_value = np.nanmean(ct * water_mask, axis=(2, 3))
        else:
            water_value = np.zeros(n_bins)

            for i in range(n_bins):
                water_value[i] = np.nanmean(ct[i, water_slice] * water_mask)

        if save_norm:
            np.save(os.path.join(directory, folder, f'water_norm_vals{append}.npy'), water_value)

    # Normalize the data
    ct = norm_hu(ct, n_bins, water_value, slice_by_slice)

    # Save the data
    if save:
        np.save(os.path.join(directory, folder, sub, 'Norm CT', f'CT_{algorithm}{append}.npy'), ct)

    # If there is contrast vials
    if contrast_mask_type:
        # Get the mask for the contrast vials
        con_path = os.path.join(directory, folder, sub, f'{contrast_mask_type}_masks.npy')
        if not os.path.exists(con_path):
            masks = create_masks(ct[-1, water_slice], mask_type=contrast_mask_type, num_rois=num_contrast_rois,
                                 folder=folder, sub=sub, directory=directory, save=save)

    return ct


def norm_hu(data, num_bins, water_value, slice_by_slice):
    """
    Normalize the EC bin to HU
    """

    data_shape = np.shape(ct)
    num_bins = data_shape[0]
    num_slices = data_shape[1]

    # Normalize to HU
    for i in range(num_bins):
        if slice_by_slice:
            for j in range(num_slices):
                data[i, j] = 1000 / water_value[i, j] * np.subtract(data[i, j], water_value[i, j])
        else:
            data[i] = 1000 / water_value[i] * np.subtract(data[i], water_value[i])

    # Get rid of any nan values
    data[np.isnan(data)] = -1000

    return data


def normalize_kedge(ct, algorithm, mask_type='contrast_200', num_contrast_rois=None, norm_slice=12, norm_val=50,
                    low_conc_norm=None, conc_vals=None, contrast='Au',
                    folder=dir_folder, sub=scan_folder, directory=data_directory, append='', save=True):
    """

    :param ct:
    :param algorithm:
    :param mask_type:
    :param num_contrast_rois:
    :param norm_slice:
    :param norm_val:
    :param low_conc_norm:
    :param conc_vals:
    :param contrast:
    :param folder:
    :param sub:
    :param directory:
    :return:
    """

    # Get the mask for the contrast vials
    con_path = os.path.join(directory, folder, sub, f'{mask_type}_masks_{contrast}.npy')
    if os.path.exists(con_path):
        masks = np.load(con_path)
    else:
        img = np.load(os.path.join(directory, folder, sub, 'CT', 'CT_FDK.npy'))[-1, 12]
        masks = create_masks(img, mask_type=mask_type, num_rois=num_contrast_rois, folder=folder, sub=sub,
                             directory=directory, save=save)

    # Normalize the K-edge data to the concentration values (in another image) that are given in conc_vals
    # or the image itself
    # These values are the image values of the highest and lowest concentrations vials before normalization
    if conc_vals is not None:
        low_conc_val_img = conc_vals[1]
        high_conc_val_img = conc_vals[0]
    else:
        # If low concentration normalization material isn't water, will use the last contrast mask
        if low_conc_norm:
            low_conc_val_img = np.nanmean(ct[norm_slice] * masks[-1])
        # Otherwise use the water mask
        else:
            # Get the mask for the water vials
            water_path = os.path.join(directory, folder, sub, f'water_mask.npy')
            if os.path.exists(water_path):
                water_mask = np.load(water_path)
            else:
                water_mask = create_masks(ct[12], mask_type='water', folder=folder, sub=sub, directory=directory)
            low_conc_val_img = np.nanmean(ct[norm_slice] * water_mask)
        # The high concentration image value will be the mean of the first contrast vial
        high_conc_val_img = np.nanmean(ct[norm_slice] * masks[0])

    # Normalize
    ct = norm_kedge(ct, low_conc_val_img, high_conc_val_img, norm_val)

    ct[ct < 0] = 0
    np.save(os.path.join(directory, folder, sub, 'Norm CT', f'CT_{algorithm}_{contrast}{append}.npy'), ct)

    return ct


def norm_kedge(data, low_conc_img, high_conc_img, high_conc_real):
    """
    Normalize the K-edge subtraction image linearly with concentration
    """

    # Normalize the K-edge data between 0 and the high concentration in real life
    data = (data - low_conc_img) / (high_conc_img - low_conc_img) * high_conc_real

    return data


if __name__ == '__main__':

    if use_norm_vals:
        water_vals = np.load(os.path.join(data_directory, dir_folder, 'water_norm_vals.npy'))
    else:
        water_vals = None


    ct = np.load(os.path.join(data_directory, dir_folder, scan_folder, 'CT', f'CT_FDK{save_note}.npy'))
    ct = np.expand_dims(ct, axis=0)
    norm_ct = normalize_ct(ct, 'FDK', water_value=water_vals, append=save_note, save_norm=save_norm_vals, save=save,
                           slice_by_slice=slice_by_slice_norm)
    #
    # k = np.load(os.path.join(data_directory, dir_folder, scan_folder, 'CT', f'K-edge_{kedge_mat}_FDK.npy'))
    # norm_k = normalize_kedge(k, 'FDK', norm_val=19.67, append='_K-edge', norm_slice=12, save=save)
    #
    # k = np.load(os.path.join(data_directory, dir_folder, scan_folder, 'CT', f'KDA_{kedge_mat}_FDK.npy'))
    # norm_k = normalize_kedge(k, 'FDK', norm_val=19.67, append='_KDA', norm_slice=12, save=save)

