#!/usr/bin/env python
# coding: utf-8

# # Medical image analysis with PyTorch
# 
# Create a deep convolutional network for an image translation task with PyTorch from scratch and train it on a subset of the IXI dataset for a T1-w to T2-w transformation.

# ## Step 5: Evaluate the results
# 
# ### Milestone 5

# In[ ]:


model.eval()
with torch.no_grad():
    out = model.forward(torch.from_numpy(t1_ex[None,None,...]).to(device)).cpu().detach().numpy()


# In[ ]:


fig,(ax1,ax2) = plt.subplots(1,2,figsize=(16,9))
ax1.imshow(out.squeeze()[:,135,:], cmap='gray'); ax1.set_title('Synthesized',fontsize=22); ax1.axis('off'); 
ax2.imshow(t2_ex[:,135,:], cmap='gray'); ax2.set_title('Truth',fontsize=22); ax2.axis('off'); 

