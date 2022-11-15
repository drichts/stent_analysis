import os
import numpy as np
import mask_functions as msk


def create_masks(img, mask_type=mask_type, num_rois=num_rois, folder=folder, sub=sub, directory=directory,
                 append='', save=True):
    """
    This function will create the desired mask(s) of specific ROI(s)
    :param mask_type: str
            The desired mask type. Types: phantom, contrast_200, contrast_600, contrast_vial_tip, mtf_patterns,
                                          mtf_contrast_square, mtf_contrast_round, water, air
    :param num_rois: int
            The number of ROIs desired. Default is None. Only needed for mask types: contrast_vial_tip and mtf_patterns
    :param folder: str
            The folder containing the desired data to create a mask for
    :param sub: str
            The specific subfolder within the folder
    :param directory: str
            LDA data directory path
    """

    img_shape = np.shape(img)
    if mask_type == 'phantom':
        # masks = msk.phantom_ROIs(img, radius=5, message_num=10)
        masks = msk.phantom_ROIs(img, radius=9, message_num=10)  # Normal
        masks[masks == 1] = 2
        masks = np.nanprod(masks, axis=0)
        masks[masks == 1] = np.nan
        masks[masks == 2] = 1
        if save:
            np.save(os.path.join(directory, folder, sub, f'phantom_mask{append}.npy'), masks)

    elif mask_type == 'water':
        masks = msk.phantom_ROIs(img, radius=6, message_num=9)
        # masks = msk.phantom_ROIs(img, radius=10, message_num=9)  # Radius = 11
        # Sum all the individual vial masks together into one mask that grabs all ROIs
        masks = np.nansum(masks, axis=0)
        masks[masks == 0] = np.nan
        if save:
            np.save(os.path.join(directory, folder, sub, f'water_mask{append}.npy'), masks)
        # np.save(os.path.join(directory, folder, sub, 'sample_masks.npy'), masks)

    elif mask_type == 'air':
        masks = msk.square_ROI(img, message_num=8)
        if save:
            np.save(os.path.join(directory, folder, sub, f'air_mask{append}.npy'), masks)

    elif mask_type == 'contrast_200':
        masks = msk.phantom_ROIs(img, radius=5, message_num=11)  # other samples
        # masks = msk.phantom_ROIs(img, radius=8, message_num=11)  # Normal sizes
        if save:
            np.save(os.path.join(directory, folder, sub, f'contrast_200_masks{append}.npy'), masks)

    elif mask_type == 'contrast_600':
        masks = msk.phantom_ROIs(img, radius=8, message_num=11)  # phantom samples HU
        # masks = msk.phantom_ROIs(img, radius=9, message_num=11)  # Normal sizes
        if save:
            np.save(os.path.join(directory, folder, sub, f'contrast_600_masks{append}.npy'), masks)

    elif mask_type == 'contrast_vial_tip':
        if num_rois:
            masks = np.zeros((num_rois, *img_shape))
            for i in range(num_rois):
                masks[i] = msk.single_pixels_mask(img)
            if save:
                np.save(os.path.join(directory, folder, sub, f'contrast_tip_masks{append}.npy'), masks)
        else:
            print(f'You must enter in the number of desired ROIs with mask type: {mask_type}')

    elif mask_type == 'mtf_patterns':
        if num_rois:
            masks = np.zeros((num_rois, *img_shape))
            for i in range(num_rois):
                masks[i] = msk.phantom_ROIs(img, radius=6)
            if save:
                np.save(os.path.join(directory, folder, sub, f'mtf_patterns{append}.npy'), masks)
        else:
            print(f'You must enter in the number of desired ROIs with mask type: {mask_type}')

    elif mask_type == 'mtf_contrast_square':
        masks = np.zeros((num_rois, *img_shape))
        for i in range(2):
            masks[i] = msk.square_ROI(img, message_num=6)
        if save:
            np.save(os.path.join(directory, folder, sub, f'mtf_contrast{append}.npy'), masks)

    elif mask_type == 'mtf_contrast_round':
        masks = np.zeros((num_rois, *img_shape))
        for i in range(2):
            masks[i] = msk.phantom_ROIs(img, radius=6)
        if save:
            np.save(os.path.join(directory, folder, sub, f'mtf_contrast{append}.npy'), masks)

    else:
        print('You have not made an accepted selection, please try again.')

    return masks


if __name__ == '__main__':

    img = np.load(os.path.join(directory, folder, sub, 'Norm CT', 'CT_FDK.npy'))[-1, 14]

    masks = create_masks(img)

