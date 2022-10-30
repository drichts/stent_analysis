import pydicom as pyd
import os
import numpy as np
import matplotlib.pyplot as plt
import mask_functions as msk
from lda.create_other_masks import create_masks
from natsort import natural_keys
from glob import glob
from lda.create_other_masks import create_masks

ct_dir = r'D:\OneDrive - University of Victoria\Research\Clinical CT'
pcd_dir = r'D:\OneDrive - University of Victoria\Research\LDA Data'

ct_folder = '22_08_09_CT_stents'
pcd_folder = '22_08_04_CT_stents'

sub_folder = 'green'

ct_path = os.path.join(ct_dir, ct_folder, '10cm_phantom', sub_folder, 'Data')
pcd_path = os.path.join(pcd_dir, pcd_folder, sub_folder + '_fast', 'Norm CT', 'CT_FDK_220pix_7slice.npy')

pcd_data = np.load(pcd_path)[5]

files = glob(os.path.join(ct_path, '*.dcm'))
files.sort(key=natural_keys)
ct_data = pyd.dcmread(files[48])
print(ct_data)
ct_data = ct_data.pixel_array[90:310, 140:360]

# Normalize CT data
# water = create_masks(ct_data, 'water', save=False)
# np.save(r'D:\OneDrive - University of Victoria\Research\Clinical CT\22_08_09_CT_stents\10cm_phantom\green\water_mask.npy', water)
water = np.load(r'D:\OneDrive - University of Victoria\Research\Clinical CT\22_08_09_CT_stents\10cm_phantom\green\water_mask.npy')

water_mean = np.nanmean(water*ct_data)

ct_data = 1000 / water_mean * np.subtract(ct_data, water_mean)

# Get rid of any nan values
ct_data[np.isnan(ct_data)] = -1000

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].axis('off')
ax[1].axis('off')

ax[0].imshow(ct_data, vmin=-400, vmax=800, cmap='gray')
ax[1].imshow(pcd_data[3], vmin=-400, vmax=800, cmap='gray')

ax[0].set_title('GE Optima 580 Simulator')
ax[1].set_title('Benchtop PCD-CT')
plt.show()
fig.savefig(r'D:\OneDrive - University of Victoria\Research\Clinical CT\22_08_09_CT_stents\10cm_phantom\green\comparison_slice_included.png', dpi=500)