import os
import tigre
import numpy as np
from tigre.utilities import sample_loader
import tigre.algorithms as algs
from ct_reconstruction.pcd_geom import PCDGeometry
import matplotlib.pyplot as plt
import time
from attenuation.filter_spectrum import filter_spectrum

# VARIABLES TO CHANGE
dir_folder = '22_10_11_CT_stents_heli'
scan_folder = 'pink_bottom'

# VARIABLES YOU MAY NEED TO CHANGE
# data_directory = r'C:\Users\drich\OneDrive - University of Victoria\Research\LDA Data'
data_directory = r'D:\OneDrive - University of Victoria\Research\LDA Data'
num_angles = 1440  # The number of projections per rotation
thresholds = [35, 52, 67, 82, 95, 120]
offset = -0.5
roll = 380  # The number of angles to roll the sinogram
recon_filter = 'shepp_logan'  # hann, hamming, cosine, shepp_logan
num_z_slices = 8
num_side_pxs = 423
kedge_bins = [1, 2]
kedge_mat = 'Ta'
filter_type = 'Al'  # Al or Cu most likely
filter_thickness = 6
append = ''

# for offset in [-0.2, -0.25, -0.3, -0.35, -0.4, -0.45, -0.5, -0.55, -0.6, -0.65, -0.7]:

path = os.path.join(r'C:\Users\drich\Documents\temp', f'{offset:0.2f}')
os.makedirs(path, exist_ok=True)
geo = PCDGeometry(h_offset=offset)
angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
angles = np.hstack([angles, angles])  # loop 2 times

# Load the sinogram data
data = np.load(os.path.join(data_directory, dir_folder, scan_folder, 'Data', 'sinogram.npy'))
data = np.roll(data, roll, axis=0)

# fig = plt.figure()
# plt.imshow(data[:, 12])
# plt.show()

# # PCD: helical
geo.offOrigin = np.zeros((angles.shape[0], 3))
geo.offOrigin[:, 0] = np.linspace(-2.481, 2.481, angles.shape[0])
geo.nVoxel = np.array((num_z_slices, num_side_pxs, num_side_pxs))
geo.sVoxel = np.array((5, 105, 105))
geo.dVoxel = geo.sVoxel / geo.nVoxel

# PCD: recon data
# FDKimg = algs.fdk(data[:, :, :, -1], geo, angles, filter=recon_filter)

#
# for i in np.arange(0, num_z_slices):
#     fig = plt.figure(figsize=(10, 10))
#     plt.imshow(FDKimg[i], vmin=0, vmax=0.03, cmap='gray')
#     plt.title(f'Slice {i+1}', fontsize=18)
#     # plt.show()
#     fig.savefig(os.path.join(path, f'{i}.png'), dpi=500)
#     # plt.pause(2)
#     plt.close()

# np.save(os.path.join(data_directory, dir_folder, scan_folder, 'CT', f'CT_FDK{append}.npy'), FDKimg)


# # # K-edge recon
att_folder = r'D:\OneDrive - University of Victoria\Research\Attenuation Data\K-edge Decomposition'
mat_att = np.loadtxt(os.path.join(att_folder, 'K-edge materials', f'{kedge_mat}.txt'))
water_att = np.loadtxt(os.path.join(att_folder, 'Background materials', 'H2O.txt'))
#
# Fetch the spectra for the weights
# spectrum = np.load(r'C:\Users\drich\OneDrive - University of Victoria\Research\Material Decomposition\Beam Spectrum\mat_decomp_spectra_120kVp.npy')
spectrum = np.load(r'D:\OneDrive - University of Victoria\Research\Material Decomposition\Beam Spectrum\mat_decomp_spectra_120kVp.npy')
spectrum[:, 1] = spectrum[:, 1] * 1E10  # Scale the weights up

# Filter the spectra
spectrum = filter_spectrum(spectrum, filter_type, filter_thickness)
spectrum = spectrum[:, 1]

# Translate from MeV to keV for the energies
mat_att[:, 0] = mat_att[:, 0] * 1000
water_att[:, 0] = water_att[:, 0] * 1000

# Look for the closest energy to the 3 thresholds
idx = []

for ei, energy in enumerate(thresholds[kedge_bins[0]:kedge_bins[1] + 2]):
    idx.append(np.argmin(np.abs(water_att[:, 0] - energy)))

low_mat = np.mean(mat_att[idx[0]:idx[1] + 1, 1])
high_mat = np.mean(mat_att[idx[1]:idx[2] + 1, 1])
low_water = np.mean(water_att[idx[0]:idx[1] + 1, 1])
high_water = np.mean(water_att[idx[1]:idx[2] + 1, 1])

# Convert the sinogram to float32 if necessary
sinogram = np.float32(data)

# K-edge decomposition
high_bin = sinogram[:, :, :, kedge_bins[1]] * low_water
low_bin = sinogram[:, :, :, kedge_bins[0]] * high_water

sinogram = (high_bin - low_bin) / ((high_mat * low_water) - (low_mat * high_water))

ct = algs.fdk(sinogram, geo, angles, filter=recon_filter)

fig_ct = plt.figure()
plt.imshow(ct[int(np.shape(ct)[0]/2)], cmap='gray', interpolation='nearest')
plt.title(f'Raw Recon K-edge {kedge_mat}')

np.save(os.path.join(data_directory, dir_folder, scan_folder, 'CT', 'KDA_FDK_HR.npy'), ct)
