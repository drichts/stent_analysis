import os
import nrrd
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID
import numpy as np
import matplotlib.pyplot as plt
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, CTImageStorage, generate_uid


def convert_npy_to_nrrd(image_set, savepath):
    """
    This function will convert a numpy array into a .nrrd set in order to use 3D slicer to create 3D volumes
    :param image_set: ndarray
            The full CT image stack you wish to convert <x, y, z>
    :param savepath: string
            The path, including the filename (with .nrrd) where you would like the data saved
    :return:
    """

    image = np.transpose(image_set, axes=(2, 1, 0))
    nrrd.write(os.path.join(savepath, f'data{append}.nrrd'), image)

