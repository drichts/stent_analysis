# authors : Guillaume Lemaitre <g.lemaitre58@gmail.com>
# license : MIT
import datetime
import os
import tempfile
from glob import glob
from natsort import natural_keys
import nrrd

import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID
import numpy as np
import matplotlib.pyplot as plt
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, CTImageStorage, generate_uid

# directory = r'C:\Users\drich\OneDrive - University of Victoria\Research'
directory = r'D:\OneDrive - University of Victoria\Research'
folder = '22_10_11_CT_stents_heli'
sub = 'red'
append = ''

bin_num = 0

def write_data_to_existing_dcm(image2d, dcm_img_file, img_num, save_path):

    print('Starting...')
    img = np.copy(image2d)
    img = np.array(img, dtype='int')
    image2d = np.array(image2d, dtype=np.uint32)

    ds = pydicom.dcmread(dcm_img_file)

    ds.BitsStored = 32
    ds.BitsAllocated = 32
    ds.SamplesPerPixel = 1
    ds.HighBit = 31

    ds.Rows = image2d.shape[0]
    ds.Columns = image2d.shape[1]
    ds.InstanceNumber = img_num + 1
    ds.DataCollectionDiameter = 105.85
    ds.ReconstructionDiameter = 105
    ds.DistanceSourcetoDetector = 578
    ds.DistanceSourcetoPatient = 322
    ds.SliceLocation = img_num * 0.184
    ds.SpacingBetweenSlices = 0
    # ds.ImagePositionPatient = rf"-52.5\-52.5\{img_num * 0.184}"
    # ds.ImageOrientationPatient = r"1\0\0\0\-1\0"
    ds.RescaleIntercept = "0"
    ds.RescaleSlope = "1"
    ds.PixelSpacing = 0.205
    ds.SliceThickness = 0.184
    ds.PixelRepresentation = 1

    print("Setting pixel data...")
    ds.PixelData = image2d.tobytes()

    ds.save_as(os.path.join(save_path, f"dicom_{img_num}.dcm"))

    print('Load file...')
    ds = pydicom.dcmread(f'out{img_num}.dcm')
    # print('Printing header...')
    # print(ds)
    fig, ax = plt.subplots(1, 2)
    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].imshow(img, vmin=-500, vmax=800, cmap='gray')
    ax[1].imshow(ds.pixel_array, vmin=-500, vmax=800, cmap='gray')
    ax[0].set_title('NDARRAY')
    ax[1].set_title('DICOM')
    plt.show()
    print(np.sum(np.abs(img - ds.pixel_array)))


def write_img_to_dcm(image2d, img_num, tot_imgs):

    print('Starting...')
    img = np.copy(image2d)
    image2d = np.array(image2d, dtype=np.uint32)

    print("Setting file meta information...")
    # Populate required values for file meta information

    meta = Dataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = meta

    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.SOPClassUID = CTImageStorage
    ds.PatientName = "Test^Firstname"
    ds.PatientID = "123456"

    ds.Modality = "CT"
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyInstanceUID = generate_uid()
    ds.FrameOfReferenceUID = generate_uid()

    ds.BitsStored = 32
    ds.BitsAllocated = 32
    ds.SamplesPerPixel = 1
    ds.HighBit = 31

    ds.ImagesInAcquisition = f"{tot_imgs}"

    ds.Rows = image2d.shape[0]
    ds.Columns = image2d.shape[1]
    ds.InstanceNumber = img_num+1

    ds.ImagePositionPatient = rf"-52.5\-52.5\{img_num*0.184}"
    ds.ImageOrientationPatient = r"1\0\0\0\-1\0"
    ds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"

    ds.RescaleIntercept = "0"
    ds.RescaleSlope = "1"
    ds.PixelSpacing = 0.205
    ds.SliceThickness = 0.184
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 1

    pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)

    print("Setting pixel data...")
    ds.PixelData = image2d.tobytes()

    ds.save_as(rf"out{img_num}.dcm")

    print('Load file...')
    ds = pydicom.dcmread(f'out{img_num}.dcm', force=True)
    print('Printing header...')
    # print(ds)
    fig, ax = plt.subplots(1, 2)
    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].imshow(img, vmin=-500, vmax=800, cmap='gray')
    ax[1].imshow(ds.pixel_array, vmin=-500, vmax=800, cmap='gray')
    ax[0].set_title('NDARRAY')
    ax[1].set_title('DICOM')


def write_2d_to_dcm(img, filepath, filename):

    # Create some temporary filenames
    suffix = '.dcm'
    filename_little_endian = tempfile.NamedTemporaryFile(suffix=suffix).name
    filename_big_endian = tempfile.NamedTemporaryFile(suffix=suffix).name

    print("Setting file meta information...")
    # Populate required values for file meta information
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = UID('1.2.840.10008.5.1.4.1.1.2')
    file_meta.MediaStorageSOPInstanceUID = UID("1.2.3")
    file_meta.ImplementationClassUID = UID("1.2.3.4")

    print("Setting dataset values...")
    # Create the FileDataset instance (initially no data elements, but file_meta
    # supplied)
    ds = FileDataset(filename_little_endian, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Add the data elements -- not trying to set all required here. Check DICOM
    # standard
    ds.PatientName = "Test^Firstname"
    ds.PatientID = "123456"

    # Set the transfer syntax
    ds.is_little_endian = True
    ds.is_implicit_VR = True

    # Set creation date/time
    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime('%Y%m%d')
    timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
    ds.ContentTime = timeStr

    # Write the pixel data to the dicom file
    img = np.asarray(img)  # Set as an ndarray
    ds.PixelData = img

    print("Writing test file", filename_little_endian)
    ds.save_as(filename_little_endian)
    print("File saved.")

    # Write as a different transfer syntax XXX shouldn't need this but pydicom
    # 0.9.5 bug not recognizing transfer syntax
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRBigEndian
    ds.is_little_endian = False
    ds.is_implicit_VR = False

    print("Writing test file as Big Endian Explicit VR", filename_big_endian)
    ds.save_as(filename_big_endian)

    # reopen the data just for checking
    for filename in (filename_little_endian, filename_big_endian):
        print('Load file {} ...'.format(filename))
        ds = pydicom.dcmread(filename)
        # ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        print(ds)
        fig = plt.figure()
        plt.axis('off')
        plt.imshow(ds.pixel_array)
        plt.pause(2)
        fig.close()

        # remove the created file
        print('Remove file {} ...'.format(filename))
        os.remove(filename)


if __name__ == '__main__':
    image = np.load(os.path.join(directory, 'LDA Data', folder, sub, 'Norm CT', f'CT_FDK{append}.npy'))
    # loadpath = os.path.join(directory, 'Clinical CT', '22_08_09_CT_stents', '10cm_phantom', 'no_stent', 'Data')
    savepath = os.path.join(directory, 'LDA Data', folder, sub, 'nrrd')

    os.makedirs(savepath, exist_ok=True)

    # files = glob(os.path.join(loadpath, '*.dcm'))
    # files.sort(key=natural_keys)

    image = np.transpose(image[bin_num], axes=(2, 1, 0))
    nrrd.write(os.path.join(savepath, f'data{append}.nrrd'), image)
    #
    # for i in range(24):
    #     # write_img_to_dcm(image[i], i, 24)
    #
    #     write_data_to_existing_dcm(image[i], files[i], i, savepath)
