import pydicom as pyd
import os
from glob import glob
from natsort import natural_keys


# directory = r'C:\Users\drich\OneDrive - University of Victoria\Research\Clinical CT'
directory = r'D:\OneDrive - University of Victoria\Research\Clinical CT'

folder = '22_09_20_CT_stents'
sub = 'green'

path = os.path.join(directory, folder, '10cm_phantom', sub, 'Data')

files = glob(os.path.join(path, '*.dcm'))
files.sort(key=natural_keys)

for f in files:
    data = pyd.dcmread(f)
    series = data['SeriesNumber'].value
    inst = data['InstanceNumber'].value
    new_name = f'{series}_{inst}'
    os.rename(f, os.path.join(path, f'{new_name}.dcm'))
    # time.sleep(100)

    # arr = data.pixel_array
    # plt.imshow(arr, cmap='gray', vmin=400, vmax=2000)
    # plt.title(f[-50:])
    # # plt.show()
    # plt.pause(0.5)

