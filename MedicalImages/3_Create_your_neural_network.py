#!/usr/bin/env python
# coding: utf-8

# # Medical image analysis with PyTorch
# 
# Create a deep convolutional network for an image translation task with PyTorch from scratch and train it on a subset of the IXI dataset for a T1-w to T2-w transformation.

# ## Step 3: Create your neural network
# 
# ### Milestone 3

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

