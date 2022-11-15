import os
import numpy as np


def convert(folder, directory=r'D:\OneDrive - University of Victoria\Research\LDA Data', append=''):
    """
    This function converts the 65s air and dark scans to 60s scans (removes the potentially bad 1st bits of data)
    :param folder: str
            The folder name within the directory in which to find the air and dark scan folders
    :param directory: str
            The filepath to the directory in which 'folder' can be found
    """
    air_folder = 'airscan_65s'
    dark_folder = 'darkscan_65s'

    save_air = os.path.join(directory, folder, 'airscan_60s', 'Data')
    save_dark = os.path.join(directory, folder, 'darkscan_60s', 'Data')

    if not os.path.exists(os.path.join(save_air, f'data{append}.npy')):
        air = np.load(os.path.join(directory, folder, air_folder, 'Data', f'data{append}.npy'))
        os.makedirs(save_air, exist_ok=True)
        air = np.sum(air[1:], axis=0)
        np.save(os.path.join(save_air, f'data{append}.npy'), air)

    if not os.path.exists(os.path.join(save_dark, f'data{append}.npy')):
        dark = np.load(os.path.join(directory, folder, dark_folder, 'Data', f'data{append}.npy'))
        os.makedirs(save_dark, exist_ok=True)
        dark = np.sum(dark[1:], axis=0)
        np.save(os.path.join(save_dark, f'data{append}.npy'), dark)
