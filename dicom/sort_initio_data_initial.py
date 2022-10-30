import matplotlib.pyplot as plt
import pydicom as pyd
import os
from glob import glob
from natsort import natural_keys

folder = '22_09_20_CT_stents'
sub = '10cm_phantom'
subsub = 'A'

# path = rf'C:\Users\drich\OneDrive - University of Victoria\Research\Clinical CT\{folder}\{sub}\{subsub}\Data'
# path = rf'D:\OneDrive - University of Victoria\Research\Clinical CT\{folder}\{sub}\{subsub}\Data'
path = rf'D:\OneDrive - University of Victoria\Research\Clinical CT\{folder}\{sub}'

o_path = os.path.join(path, subsub)

# data = pyd.dcmread(os.path.join(path, 'CT.1.2.840.113619.2.278.3.34236641.67.1659565077.28.208.dcm'))
#
# print(data)
#
# arr = data.pixel_array
#
#
# plt.imshow(arr, cmap='gray', vmin=700, vmax=1200)
# plt.axis('off')
# plt.show()
# plt.savefig(rf'D:\OneDrive - University of Victoria\Files\Grad School\Meetings\22-08-09_PCD\{sub}_CT.png', dpi=500)

data = pyd.dcmread(r'D:\OneDrive - University of Victoria\Research\Clinical CT\22_10_19_CT_stents\10cm_phantom\none_stnd\Data\3.dcm')
print(data)
# files = glob(os.path.join(o_path, '*'))
# # print(files)
# files.sort(key=natural_keys)
#
# # print(files)
# i = 0
# for f in files:
#     data = pyd.dcmread(f)
#     # print(f[-4:])
#     org = data['AcquisitionTime'].value
#     # print()
#     filename = f.split(o_path)[1][1:]
#     # print(filename)
#     temppath = os.path.join(path, org)
#     #
#     if not os.path.exists(temppath):
#         os.makedirs(temppath, exist_ok=True)
#
#     os.rename(f, os.path.join(temppath, filename + '.dcm'))

