# # Medical image analysis with PyTorch
#
# Create a deep convolutional network for an image translation task with
# PyTorch from scratch and train it on a subset of the IXI dataset for a
# T1-w to T2-w transformation.


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
import torchsummary

import LocalTransforms

def reportVersions() :
    print('numpy version: {}'.format(np.__version__))
    from matplotlib import __version__ as mplver
    print('matplotlib version: {}'.format(mplver))
    print(f'pytorch version: {torch.__version__}')
    print(f'torchvision version: {torchvision.__version__}')
    pv = sys.version_info
    print('python version: {}.{}.{}'.format(pv.major, pv.minor, pv.micro))

    # Check GPU(s)
    os.system('nvidia-smi')


def setSeeds() :
    # Set seeds for reproducibility
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# ## Step 1: Training and validation data setup
#

def checkData( train_dir ) :
    train_dir = 'F:\\Home\\Projects\\LiveProjects\\MedicalImages\\small'
    t1_dir = os.path.join(train_dir, 't1')
    t2_dir = os.path.join(train_dir, 't2')


    t1_fns = glob(os.path.join(t1_dir, '*.nii*'))
    t2_fns = glob(os.path.join(t2_dir, '*.nii*'))
    assert len(t1_fns) == len(t2_fns) and len(t1_fns) != 0
    print(f'Found {len(t1_fns)} T1 files and {len(t2_fns)} T2 files.')


    # ### Milestone 1
    #
    # Look at an axial view of the source T1-weighted (T1-w) and target T2-weighted (T2-w) images.


    t1_ex, t2_ex = nib.load(t1_fns[0]).get_data(), nib.load(t2_fns[0]).get_data()
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(16,9))
    ax1.imshow(t1_ex[:,115,:], cmap='gray'); ax1.set_title('T1',fontsize=22); ax1.axis('off');
    ax2.imshow(t2_ex[:,115,:], cmap='gray'); ax2.set_title('T2',fontsize=22); ax2.axis('off');
    plt.show()

    print('End of Milestone 1')

# ## Step 2: Datasets and transforms

def glob_imgs(path: str, ext='*.nii*') -> List[str]:
    """ grab all `ext` files in a directory and sort them for consistency """
    fns = sorted(glob(os.path.join(path, ext)))
    return fns


class NiftiDataset(Dataset):
    """
    create a dataset class in PyTorch for reading NIfTI files

    Args:
        source_dir (str): path to source images
        target_dir (str): path to target images
        transform (Callable): transform to apply to both source and target images
        preload (bool): load all data when initializing the dataset
    """

    def __init__(self, source_dir:str, target_dir:str, transform:Optional[Callable]=None, preload:bool=True):
        self.source_dir, self.target_dir = source_dir, target_dir
        self.source_fns, self.target_fns = glob_imgs(source_dir), glob_imgs(target_dir)
        self.transform = transform
        self.preload = preload
        if len(self.source_fns) != len(self.target_fns) or len(self.source_fns) == 0:
            raise ValueError(f'Number of source and target images must be equal and non-zero')
        if preload:
            self.imgs = [(nib.load(s).get_data().astype(np.float32),
                          nib.load(t).get_data().astype(np.float32))
                         for s, t in zip(self.source_fns, self.target_fns)]

    def __len__(self):
        return len(self.source_fns)

    def __getitem__(self, idx:int):
        ### print(f'Check 300 getitem called idx={idx}')
        if not self.preload:
            src_fn, tgt_fn = self.source_fns[idx], self.target_fns[idx]
            sample = (nib.load(src_fn).get_data(), nib.load(tgt_fn).get_data())
        else:
            sample = self.imgs[idx]
        if self.transform is not None:
            ### print(f'Check 310 Before transform to shape {sample[0].shape} and {sample[1].shape}')
            sample = self.transform(sample)
            ### print(f'Check 320 After transform to shape {sample[0].shape} and {sample[1].shape}')
        return sample



# ## Step 3: Create your neural network

def conv(i,o):
    return (nn.Conv3d(i,o,3,padding=1,bias=False),
            nn.BatchNorm3d(o),
            nn.ReLU(inplace=True))

def unet_block(i,m,o):
    return nn.Sequential(*conv(i,m),*conv(m,o))

class Unet(nn.Module):
    def __init__(self, s=32):
        super().__init__()
        self.start = unet_block(1,s,s)
        self.down1 = unet_block(s,s*2,s*2)
        self.down2 = unet_block(s*2,s*4,s*4)
        self.bridge = unet_block(s*4,s*8,s*4)
        self.up2 = unet_block(s*8,s*4,s*2)
        self.up1 = unet_block(s*4,s*2,s)
        self.final = nn.Sequential(*conv(s*2,s),nn.Conv3d(s,1,1))

    def forward(self,x):
        ### x=x.unsqueeze(1)
        print(f'Check 400 x.shape={x.shape}')
        r = [self.start(x)]
        print(f'Check 410 len(r)={len(r)}')
        r.append(self.down1(F.max_pool3d(r[-1],2)))
        r.append(self.down2(F.max_pool3d(r[-1],2)))
        print(f'Check 420 len(r)={len(r)}')
        x = F.interpolate(self.bridge(F.max_pool3d(r[-1],2)),size=r[-1].shape[2:])
        x = F.interpolate(self.up2(torch.cat((x,r[-1]),dim=1)),size=r[-2].shape[2:])
        x = F.interpolate(self.up1(torch.cat((x,r[-2]),dim=1)),size=r[-3].shape[2:])
        x = self.final(torch.cat((x,r[-3]),dim=1))
        return x


# ## Step 4: Train the network


def trainModel( train_dir ) :

    valid_split = 0.1
    batch_size = 16
    n_jobs = 12
    n_epochs = 50


    ### tfms = Compose([LocalTransforms.RandomCrop3D((128,128,32)), LocalTransforms.ToTensor()])
    tfms = Compose([LocalTransforms.RandomCrop3D((90, 90, 32)), LocalTransforms.ToTensor()])

    # set up training and validation data loader for nifti images
    t1_dir = os.path.join(train_dir, 't1')
    t2_dir = os.path.join(train_dir, 't2')

    dataset = NiftiDataset(t1_dir, t2_dir, tfms, preload=False)  # set preload=False if you have limited CPU memory
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(valid_split * num_train)
    valid_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(valid_idx))
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size,
                              num_workers=n_jobs, pin_memory=True)
    valid_loader = DataLoader(dataset, sampler=valid_sampler, batch_size=batch_size,
                              num_workers=n_jobs, pin_memory=True)

    assert torch.cuda.is_available()
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True

    model = Unet()
    nElementsList = [p.numel() for p in model.parameters()]
    nElements = sum( nElementsList )
    print(f'Total number of model parameters = {nElements}')
    print(f'Number of parameters by layer = {nElementsList}')

    #model.load_state_dict(torch.load('trained.pth'));

    model.cuda(device=device)

    print('=================================')
    torchsummary.summary( model, (1,90,90,32) )
    print('=================================')

    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=1e-6)
    criterion = nn.SmoothL1Loss()  #nn.MSELoss()

    train_losses, valid_losses = [], []
    n_batches = len(train_loader)
    for t in range(1, n_epochs + 1):
        # training
        t_losses = []
        model.train(True)
        for i, (src, tgt) in enumerate(train_loader):
            print(f'Check 100 i={i}')
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            out = model(src)
            loss = criterion(out, tgt)
            t_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        train_losses.append(t_losses)

        # validation
        v_losses = []
        model.train(False)
        with torch.set_grad_enabled(False):
            for src, tgt in valid_loader:
                src, tgt = src.to(device), tgt.to(device)
                out = model(src)
                loss = criterion(out, tgt)
                v_losses.append(loss.item())
            valid_losses.append(v_losses)

        if not np.all(np.isfinite(t_losses)):
            raise RuntimeError('NaN or Inf in training loss, cannot recover. Exiting.')
        log = f'Epoch: {t} - Training Loss: {np.mean(t_losses):.2e}, Validation Loss: {np.mean(v_losses):.2e}'
        print(log)


    torch.save(model.state_dict(), 'trained.pth')

    # ## Step 5: Evaluate the results

    model.eval()

    t1_fns, t2_fns =  glob_imgs(t1_dir), glob_imgs(t2_dir)
    t1_ex, t2_ex = nib.load(t1_fns[0]).get_data(), nib.load(t2_fns[0]).get_data()

    with torch.no_grad():
        out = model.forward(torch.from_numpy(t1_ex[None,None,...]).to(device)).cpu().detach().numpy()


    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(16,9))
    ax1.imshow(out.squeeze()[:,135,:], cmap='gray'); ax1.set_title('Synthesized',fontsize=22); ax1.axis('off');
    ax2.imshow(t2_ex[:,135,:], cmap='gray'); ax2.set_title('Truth',fontsize=22); ax2.axis('off');


def Main():
    reportVersions()
    setSeeds()
    train_dir = 'F:\\Home\\Projects\\LiveProjects\\MedicalImages\\small'
    ### checkData( train_dir )

    trainModel(train_dir)

if __name__ == '__main__':
    Main()


