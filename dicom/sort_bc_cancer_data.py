import matplotlib.pyplot as plt
import pydicom as pyd
import os
from glob import glob
from natsort import natural_keys

directory = r'D:\OneDrive - University of Victoria\Research\Clinical CT\22_10_19_CT_stents'
folder = 'drichtstents2210'

files = glob(os.path.join(directory, folder, '*.dcm'))

for file in files:
    data = pyd.dcmread(file)
    s = data['SeriesNumber'].value  # This is the series number in my notebook (I think)
    savepath = os.path.join(directory, f'{s}', 'Data')
    os.makedirs(savepath, exist_ok=True)
    z = data['InstanceNumber'].value
    os.rename(file, os.path.join(savepath, f'{z}.dcm'))

