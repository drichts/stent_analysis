# 3D volume figure

import os
import numpy as np
import pydicom as pyd
import matplotlib.pyplot as plt
from glob import glob
import pyvista as pv

# Define the directory (which computer you're on)
# directory = r'C:\Users\drich - University of Victoria\Research'
directory = r'D:\OneDrive - University of Victoria\Research'

# Type of acquisition folder
clin_dir = 'Clinical CT'
pcd_dir = 'LDA Data'

# Specific folder defining the day
clin_folder = '22_08_09_CT_stents'
pcd_folder = '22_09_07_CT_stents'

clin_path = os.path.join(directory, clin_dir, clin_folder, '10cm_phantom')
pcd_path = os.path.join(directory, pcd_dir, pcd_folder)

# Grab the PCD vtk stent data
red_pcd = pv.read(os.path.join(pcd_path, 'red_mid', 'nrrd', 'stent_transposed.vtk'))
purple_pcd = pv.read(os.path.join(pcd_path, 'purple_mid', 'nrrd', 'stent_transposed.vtk'))
pink_pcd = pv.read(os.path.join(pcd_path, 'pink_mid', 'nrrd', 'stent_transposed.vtk'))

# Grab the clinical vtk stent data
red_clin = pv.read(os.path.join(clin_path, 'red', 'stent_HD_clip.vtk'))
purple_clin = pv.read(os.path.join(clin_path, 'purple', 'stent_HD_clip.vtk'))
pink_clin = pv.read(os.path.join(clin_path, 'pink', 'stent_HD_clip.vtk'))

# # Set up a pyvista sublotter
# pl = pv.Plotter(shape=(2, 3))
# actor = pl.add_mesh(red_clin, smooth_shading=True)
# # pl.add_title(r'Prot$\acute{e}$g$\acute{e}$')
# pl.subplot(0, 1)
# actor = pl.add_mesh(purple_clin, smooth_shading=True)
# # pl.add_title('Precise')
# pl.subplot(0, 2)
# actor = pl.add_mesh(pink_clin, smooth_shading=True)
# # pl.add_title('S.M.A.R.T. Control')
# pl.subplot(1, 0)
# actor = pl.add_mesh(red_pcd, smooth_shading=True)
# pl.subplot(1, 1)
# actor = pl.add_mesh(purple_pcd, smooth_shading=True)
# pl.subplot(1, 2)
# actor = pl.add_mesh(pink_pcd, smooth_shading=True)
# pl.show()

# pl.close()
#
pl = pv.Plotter()
_ = pl.add_mesh(red_clin, smooth_shading=True)
_ = pl.set_background('gray')
pl.save_graphic(os.path.join(r'D:\OneDrive - University of Victoria\Files\Grad School\Manuscripts\Stents\Figures', 'red_clin_3D.svg'))
pl.show()
pl.close()
#
# pl = pv.Plotter()
# _ = pl.add_mesh(purple_clin, smooth_shading=True)
# pl.save_graphic(os.path.join(r'D:\OneDrive - University of Victoria\Files\Grad School\Manuscripts\Stents\Figures', 'purple_clin_3D.svg'))
# pl.close()
#
# pl = pv.Plotter()
# _ = pl.add_mesh(pink_clin, smooth_shading=True)
# pl.save_graphic(os.path.join(r'D:\OneDrive - University of Victoria\Files\Grad School\Manuscripts\Stents\Figures', 'pink_clin_3D.svg'))
# pl.close()
#
# pl = pv.Plotter()
# _ = pl.add_mesh(red_pcd, smooth_shading=True)
# pl.save_graphic(os.path.join(r'D:\OneDrive - University of Victoria\Files\Grad School\Manuscripts\Stents\Figures', 'red_pcd_3D.svg'))
# pl.close()
#
# pl = pv.Plotter()
# _ = pl.add_mesh(purple_pcd, smooth_shading=True)
# pl.save_graphic(os.path.join(r'D:\OneDrive - University of Victoria\Files\Grad School\Manuscripts\Stents\Figures', 'purple_pcd_3D.svg'))
# pl.close()
#
# pl = pv.Plotter()
# _ = pl.add_mesh(pink_pcd, smooth_shading=True)
# pl.save_graphic(os.path.join(r'D:\OneDrive - University of Victoria\Files\Grad School\Manuscripts\Stents\Figures', 'pink_pcd_3D.svg'))
# pl.close()

