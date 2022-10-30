import numpy as np
import os
import matplotlib.pyplot as plt

folder = 'pink_bottom'

# data = np.load(rf'D:\OneDrive - University of Victoria\Research\LDA Data\22_10_11_CT_stents_heli\{folder}\Data\data.npy')[:, :, :, -1]
# air = np.load(r'D:\OneDrive - University of Victoria\Research\LDA Data\22_10_11_CT_stents_heli\airscan_60s\Data\data.npy')[:, :, -1]/480
#
# data = np.log(air) - np.log(data)
#
# data = np.roll(data[20:-20], 200, axis=0)
# plt.imshow(data[150:250, 12], vmin=0, vmax=2)
# plt.show()


sino = np.load(rf'D:\OneDrive - University of Victoria\Research\LDA Data\22_10_11_CT_stents_heli\{folder}\Data\sinogram.npy')[:, :, :, -1]
sino = np.roll(sino[5:-35], 200, axis=0)

plt.imshow(sino[150:250, 12], vmin=0, vmax=2)
plt.show()