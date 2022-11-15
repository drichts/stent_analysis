import pydicom as pyd
import os
from glob import glob
from natsort import natural_keys


def sort_dicom_files(path):
    """
    This function will take the folder path leading to where all the dicom files of a specific CT scan are and sort
    them in order of slice. NOTE: the folder should only contain files from 1 CT scan and no other dicom files
    It renames the files based on their series and instance number from the header
    :param path: string
            The full folder path
    """

    # Sort the files
    files = glob(os.path.join(path, '*.dcm'))
    files.sort(key=natural_keys)

    # Rename each file based on its series and instance number
    for f in files:
        data = pyd.dcmread(f)
        series = data['SeriesNumber'].value
        inst = data['InstanceNumber'].value
        new_name = f'{series}_{inst}'
        os.rename(f, os.path.join(path, f'{new_name}.dcm'))

