import matplotlib.pyplot as plt
import pydicom as pyd
import os
from glob import glob
from natsort import natural_keys

folder = '22_10_19_CT_stents'
sub = 'purple_boneplus'

# path = rf'C:\Users\drich\OneDrive - University of Victoria\Research\Clinical CT\{folder}\10cm_phantom\{sub}\Data'
path = rf'D:\OneDrive - University of Victoria\Research\Clinical CT\{folder}\10cm_phantom\{sub}\Data'

files = glob(os.path.join(path, '*'))
files.sort(key=natural_keys)

# print(files)

#
# f1 = pyd.dcmread(rf'D:\OneDrive - University of Victoria\Research\Clinical CT\{folder}\10cm_phantom\pink_stnd\Data\88.dcm')
# f2 = pyd.dcmread(rf'D:\OneDrive - University of Victoria\Research\Clinical CT\{folder}\10cm_phantom\pink_detail\Data\88.dcm')
# f3 = pyd.dcmread(rf'D:\OneDrive - University of Victoria\Research\Clinical CT\{folder}\10cm_phantom\pink_bone\Data\88.dcm')
# f4 = pyd.dcmread(rf'D:\OneDrive - University of Victoria\Research\Clinical CT\{folder}\10cm_phantom\pink_boneplus\Data\88.dcm')
# f5 = pyd.dcmread(rf'D:\OneDrive - University of Victoria\Research\Clinical CT\{folder}\10cm_phantom\pink_edge\Data\88.dcm')
#
# fig, ax = plt.subplots(2, 3)
# ax[0, 0].imshow(f1.pixel_array - 1024, vmin=-500, vmax=2000, cmap='gray')
# ax[0, 1].imshow(f2.pixel_array - 1024, vmin=-500, vmax=2000, cmap='gray')
# ax[0, 2].imshow(f3.pixel_array - 1024, vmin=-500, vmax=2000, cmap='gray')
# ax[1, 0].imshow(f4.pixel_array - 1024, vmin=-500, vmax=2000, cmap='gray')
# ax[1, 1].imshow(f5.pixel_array - 1024, vmin=-500, vmax=2000, cmap='gray')
# plt.show()
fig = plt.figure()
for i, f in enumerate(files[80:]):
    data = pyd.dcmread(f)

    arr = data.pixel_array - 1024

    plt.imshow(arr, cmap='gray', vmin=-500, vmax=2000)
    plt.title(80 + i)
    plt.pause(0.25)
    # fig.savefig(rf'C:\Users\drich\OneDrive - University of Victoria\Research\Clinical CT\{folder}\{sub}\CT_{i}.png', dpi=500)
    # plt.pause(1)


