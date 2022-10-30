import pydicom as pyd
import os
from glob import glob
from natsort import natural_keys

directory = r'C:\Users\drich\OneDrive - University of Victoria\Research\Clinical CT\22_09_20_CT_stents\10cm_phantom\other_dicom\121740'

files = glob(os.path.join(os.path.join(directory, '*.dcm')))

data = pyd.dcmread(files[0])
print(data)

