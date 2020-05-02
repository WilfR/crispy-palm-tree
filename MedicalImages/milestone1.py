import nibabel as nib
from matplotlib import pyplot as plt
from matplotlib.colors import NoNorm
import numpy as np


t1image = nib.load('F:/Home/Projects/LiveProjects/MedicalImages/small/t1/IXI160-HH-1637-T1_fcm.nii.gz')
t2image = nib.load('F:/Home/Projects/LiveProjects/MedicalImages/small/t2/IXI160-HH-1637-T2_reg_fcm.nii.gz')


t1image_data = t1image.get_fdata()
t2image_data = t2image.get_fdata()


sliceNumber  = 35

t1slice = t1image_data[:,:,sliceNumber]
t2slice = t2image_data[:,:,sliceNumber]

print(t1image_data.shape)
print(t2image_data.shape)
print(f'T1 min={np.amin(t1slice)} max={np.amin(t1slice)}')
print(f'T2 min={np.amin(t2slice)} max={np.amin(t2slice)}')


plt.subplot(1,2,1)
plt.imshow(np.rot90(t1slice), interpolation='nearest', cmap='gray')
plt.subplot(1,2,2)
plt.imshow(np.rot90(t2slice), interpolation='nearest', cmap='gray')

plt.show()
