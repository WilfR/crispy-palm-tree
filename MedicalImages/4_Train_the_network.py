#!/usr/bin/env python
# coding: utf-8

# # Medical image analysis with PyTorch
# 
# Create a deep convolutional network for an image translation task with PyTorch from scratch and train it on a subset of the IXI dataset for a T1-w to T2-w transformation.

# ## Step 4: Train the network

# In[ ]:





# In[ ]:


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
        r = [self.start(x)]
        r.append(self.down1(F.max_pool3d(r[-1],2)))
        r.append(self.down2(F.max_pool3d(r[-1],2)))
        x = F.interpolate(self.bridge(F.max_pool3d(r[-1],2)),size=r[-1].shape[2:])
        x = F.interpolate(self.up2(torch.cat((x,r[-1]),dim=1)),size=r[-2].shape[2:])
        x = F.interpolate(self.up1(torch.cat((x,r[-2]),dim=1)),size=r[-3].shape[2:])
        x = self.final(torch.cat((x,r[-3]),dim=1))
        return x


# In[1]:


valid_split = 0.1
batch_size = 16
n_jobs = 12
n_epochs = 50


# In[2]:


tfms = Compose([RandomCrop3D((128,128,32)), ToTensor()])

# set up training and validation data loader for nifti images
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


# ### Milestone 4

# In[ ]:


assert torch.cuda.is_available()
device = torch.device('cuda:0')
torch.backends.cudnn.benchmark = True


# In[ ]:


model = Unet()


# In[ ]:


#model.load_state_dict(torch.load('trained.pth'));


# In[ ]:


model.cuda(device=device)
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=1e-6)
criterion = nn.SmoothL1Loss()  #nn.MSELoss()


# In[ ]:


train_losses, valid_losses = [], []
n_batches = len(train_loader)
for t in range(1, n_epochs + 1):
    # training
    t_losses = []
    model.train(True)
    for i, (src, tgt) in enumerate(train_loader):
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


# In[ ]:


torch.save(model.state_dict(), 'trained.pth')

