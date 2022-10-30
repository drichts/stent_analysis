import os
import numpy as np

data_directory = r'D:\OneDrive - University of Victoria\Research\LDA Data'
data_folder = '22-01-31_2D_shortribs'
airfolder = 'airscan_2.0Al_65s'
darkfolder = 'darkscan_65s'


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


if __name__ == '__main__':

    save_air = os.path.join(data_directory, data_folder, 'airscan_2.0Al_60s', 'Data')
    # save_dark = os.path.join(data_directory, data_folder, 'darkscan_60s', 'Data')

    air = np.load(os.path.join(data_directory, data_folder, airfolder, 'Data', f'data.npy'))
    os.makedirs(save_air, exist_ok=True)
    air = np.sum(air[1:], axis=0)
    np.save(os.path.join(save_air, f'data.npy'), air)

    # dark = np.load(os.path.join(data_directory, data_folder, darkfolder, 'Data', f'data.npy'))
    # os.makedirs(save_dark, exist_ok=True)
    # dark = np.sum(dark[1:], axis=0)
    # np.save(os.path.join(save_dark, f'data.npy'), dark)
