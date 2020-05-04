#!/usr/bin/env python
# coding: utf-8

# # Medical image analysis with PyTorch
# 
# Create a deep convolutional network for an image translation task with PyTorch from scratch and train it on a subset of the IXI dataset for a T1-w to T2-w transformation.

# ### Setup notebook

# In[ ]:


from typing import Callable, List, Optional, Tuple, Union

from glob import glob
import os
import random
import sys

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision.transforms import Compose


#  Support in-notebook plotting

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# Report versions

# In[ ]:


print('numpy version: {}'.format(np.__version__))
from matplotlib import __version__ as mplver
print('matplotlib version: {}'.format(mplver))
print(f'pytorch version: {torch.__version__}')
print(f'torchvision version: {torchvision.__version__}')


# In[ ]:


pv = sys.version_info
print('python version: {}.{}.{}'.format(pv.major, pv.minor, pv.micro))


# Reload packages where content for package development

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# Check GPU(s)

# In[ ]:


get_ipython().system('nvidia-smi')


# Set seeds for reproducibility

# In[ ]:


seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


# ## Step 1: Training and validation data setup
# 
# Get the location of the training (and validation) data

# In[ ]:


train_dir = 'set this to full path to training data'
t1_dir = os.path.join(train_dir, 't1')
t2_dir = os.path.join(train_dir, 't2')


# In[ ]:


t1_fns = glob(os.path.join(t1_dir, '*.nii*'))
t2_fns = glob(os.path.join(t2_dir, '*.nii*'))
assert len(t1_fns) == len(t2_fns) and len(t1_fns) != 0


# ### Milestone 1
# 
# Look at an axial view of the source T1-weighted (T1-w) and target T2-weighted (T2-w) images.

# In[ ]:


t1_ex, t2_ex = nib.load(t1_fns[0]).get_data(), nib.load(t2_fns[0]).get_data()
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(16,9))
ax1.imshow(t1_ex[:,135,:], cmap='gray'); ax1.set_title('T1',fontsize=22); ax1.axis('off'); 
ax2.imshow(t2_ex[:,135,:], cmap='gray'); ax2.set_title('T2',fontsize=22); ax2.axis('off'); 

