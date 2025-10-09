#!/usr/bin/env python
# coding: utf-8

# 
# # <strong><center> Sarah Eldeen</center></strong>

# # CNN on Bead Images
# For this homework, I will be using data I collected in my research lab. The data consists of microbead images within a hydrogel environement or water. Due to diffraction of light, the bead in focus will appear as white and the out of focus as black or gray. One of my analysis goals is to track beads in XYZ. However, my tracking algorithms depends on the center of mass calculation. Which means that only "white" beads can be tracked. Therefore, my two output categories will be: "Good" for when the beads are most in focus and "Bad" for anything else. 

# ## Perform standard imports

# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models # add models to the list for AlexNet pretrained model
from torchvision.utils import make_grid

import os
import numpy as np
from patchify import patchify, unpatchify
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Define transforms

# In[3]:


# I chose my images size to be 300x300
train_transform = transforms.Compose([
        transforms.RandomRotation(10),      # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.Resize(300),             # resize shortest side to 300 pixels
        transforms.CenterCrop(300),         # crop longest side to 300 pixels at center
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

test_transform = transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


# ## Data loading
# I will be using a built-in torchvision dataset tool called <strong>ImageFolder</strong></tt></a>.

# In[4]:


root = '../Data/BEAD' #path

train_data = datasets.ImageFolder(os.path.join(root, 'train'), transform=train_transform)
test_data = datasets.ImageFolder(os.path.join(root, 'test'), transform=test_transform)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True) # experiment with batch size
test_loader = DataLoader(test_data, batch_size=10, shuffle=True)

class_names = train_data.classes

print(class_names)
print(f'Training images available: {len(train_data)}')
print(f'Testing images available:  {len(test_data)}')


# ## Display a batch of images

# In[5]:


# Grab the first batch of 5 images
for images,labels in train_loader: 
    break

# Print the labels
print('Label:', labels.numpy())
print('Class:', *np.array([class_names[i] for i in labels]))

im = make_grid(images, nrow=5)  # the default nrow is 8

# Inverse normalize the images
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)
im_inv = inv_normalize(im)

# Print the images
plt.figure(figsize=(12,4))
plt.imshow(np.transpose(im_inv.numpy(), (1, 2, 0)));


# ## Define the model

# In[6]:


class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 3, 1) #3 color channels, create 6 filters out of them. kernel size is 3x3 stride = 1
        self.conv2 = nn.Conv2d(3, 16, 3, 1) #must match. expand filters
        self.fc1 = nn.Linear(73*73*16, 120) #reducing that with fully connected layers #experiment with this
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        #self.fc4 = nn.Linear(84,35) add something in between

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 73*73*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)


# ### Instantiate the model, define loss and optimization functions

# In[7]:


CNNmodel = ConvolutionalNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(CNNmodel.parameters(), lr=0.001)
CNNmodel


# ## Train the model

# In[8]:


import time
start_time = time.time()

epochs = 4

train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    trn_corr = 0
    tst_corr = 0
    
    # Run the training batches
    for b, (X_train, y_train) in enumerate(train_loader):
        b+=1
        
        # Apply the model
        y_pred = CNNmodel(X_train)
        loss = criterion(y_pred, y_train)
 
        # Tally the number of correct predictions
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr
        
        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print interim results
        if b%100 == 0:

             print(f'epoch: {i:2}  batch: {b:4} [{10*b:6}/{len(train_data)}]  loss: {loss.item():10.8f}  \
accuracy: {trn_corr.item()*100/(10*b):7.3f}%')

 
    train_losses.append(loss)
    train_correct.append(trn_corr)

    # Run the testing batches
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):

            # Apply the model
            y_val = CNNmodel(X_test)

            # The number of correct predictions
            predicted = torch.max(y_val.data, 1)[1] 
            tst_corr += (predicted == y_test).sum()

    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)

print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed


# ## Evaluate model performance

# In[12]:


plt.plot(train_losses, label='training loss')
plt.plot(test_losses, label='validation loss')
plt.title('Loss at the end of each epoch')
plt.legend();


# In[13]:


plt.plot([t/(len(train_data)/10) for t in train_correct], label='training accuracy')
plt.plot([t/(len(test_data)/10) for t in test_correct], label='validation accuracy')
plt.title('Accuracy at the end of each epoch')
plt.legend();


# ## Comments on results
# Overall, the loss plot decreases when it reaches 4 epochs, which is expected. The accuracy seems to be as expected as well. Both lines reach the same value when the number of epochs is approperiate. 

# In[14]:


print(test_correct)
print(f'Test accuracy: {test_correct[-1].item()*100/len(test_data):.3f}%')


# ## It is performing better than a random guess (50%)

# ## Save the trained model

# In[37]:


torch.save(CNNmodel.state_dict(), 'CNNModel.pt')


# ## Try a pretrained model
# <strong>torchvision.models</strong></tt></a>:
# <ul>
# <li><a href="https://arxiv.org/abs/1404.5997">AlexNet</a></li>
# </ul>
# My task is to reduce the output of the fully connected layers from 1000 categories to just 2.
# 
# To access the model, I will use:<br>
# <pre>resnet18 = models.resnet18()</pre>
# I will also obtain a pre-trained model by passing pretrained=True:<br>
# <pre>resnet18 = models.resnet18(pretrained=True)</pre>
# All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
# 
# I already transformend my images to the correct format to be able to test this model. 

# In[16]:


AlexNetmodel = models.alexnet(pretrained=True)
AlexNetmodel


# ## Freeze feature parameters

# In[17]:


for param in AlexNetmodel.parameters():
    param.requires_grad = False


# ## Modify the classifier

# In[18]:


AlexNetmodel.classifier = nn.Sequential(nn.Linear(9216, 1024),
                                 nn.ReLU(),
                                 nn.Dropout(0.4), # just use 40%
                                 nn.Linear(1024, 2), 
                                 nn.LogSoftmax(dim=1))
AlexNetmodel


# ## Define loss function & optimizer

# In[19]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(AlexNetmodel.classifier.parameters(), lr=0.001)


# ## Train the model
# We only need to run one epoch.

# In[20]:


import time
start_time = time.time()

epochs = 1

train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    trn_corr = 0
    tst_corr = 0
    
    # Run the training batches
    for b, (X_train, y_train) in enumerate(train_loader):

        b+=1
        
        # Apply the model
        y_pred = AlexNetmodel(X_train)
        loss = criterion(y_pred, y_train)
 
        # Tally the number of correct predictions
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr
        
        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print interim results
        if b%100 == 0:
               print(f'epoch: {i:2}  batch: {b:4} [{10*b:6}/{len(train_data)}]  loss: {loss.item():10.8f}  \
accuracy: {trn_corr.item()*100/(10*b):7.3f}%')


    train_losses.append(loss)
    train_correct.append(trn_corr)

    # Run the testing batches
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):


            # Apply the model
            y_val = AlexNetmodel(X_test)

            # Tally the number of correct predictions
            predicted = torch.max(y_val.data, 1)[1] 
            tst_corr += (predicted == y_test).sum()

    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)

print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed


# In[21]:


print(test_correct)
print(f'Test accuracy: {test_correct[-1].item()*100/len(test_data):.3f}%')


# ### My model performed better than AlexNet pretrained model. Maybe because the data I am using has "less complexity" than the ones used to train this model.

# ## Run a new image through the model

# In[39]:


x = 15
im = inv_normalize(test_data[x][0])
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)));
# CNN Model Prediction:

CNNmodel.eval()
with torch.no_grad(): # dont update any weights or biases
    new_pred = CNNmodel(test_data[x][0].view(1,3,300,300)).argmax()
print(f'CNNmodel     Predicted value: {new_pred.item()} {class_names[new_pred.item()]}')
# AlexNet Model Prediction:

AlexNetmodel.eval()
with torch.no_grad():
    new_pred = AlexNetmodel(test_data[x][0].view(1,3,300,300)).argmax()
print(f'AlexNetModel Predicted value: {new_pred.item()} {class_names[new_pred.item()]}')


# # Test on Large Image
# This section was very hard to accomplish. I had to create patches from the raw image and fix its size. Then, I had to figure out how to apply my model on the patches and draw corresponding rectanngles.

# In[53]:


root = '../untitledfolder' #path
# input image
img = Image.open("../untitledfolder/testimage_2000nm.jpeg")
plt.imshow(img)
# display the image
image = np.asarray(img)
# display the image
print(image.shape)
# splitting the image into patches
image_height, image_width, channel_count = image.shape
patch_height, patch_width, step = 60, 60,20
patch_shape = (patch_height, patch_width, channel_count)
patches = patchify(image, patch_shape, step=step)
print(patches.shape)

# processing each patch
output_patches = np.empty(patches.shape).astype(np.uint8)
CNNresults = np.empty(patches.shape)
ANresults = np.empty(patches.shape)

for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        patch = patches[i, j, 0]


# In[56]:


randompatch = Image.fromarray(patches[1, 2, 0])
plt.imshow(randompatch) 


# In[57]:


CNNresults = []
ANresults = []
matchcol = []
matchrow = []
a = 1
index1 = 0
index2 = 0

for index1 in range(patches.shape[0]):
    for index2 in range(patches.shape[1]):
        select = Image.fromarray(patches[index1, index2, 0])
        x = test_transform(select)
    # CNN Model Prediction:
        CNNmodel.eval()
        with torch.no_grad(): # dont update any weights or biases
            new_pred = CNNmodel(x.view(1,3,300,300)).argmax()
            #print(f'CNNmodel     Predicted value: {new_pred.item()} {class_names[new_pred.item()]}')
            if new_pred.item() == a:
                matchrow.append(index2)
                matchcol.append(index1)
        index2 += 1
    index1 += 1


# In[61]:


print(matchcol[2],matchrow[2])
randompatch = Image.fromarray(patches[0, 40, 0])
plt.imshow(randompatch) 


# In[68]:


import matplotlib.pyplot as plt
#import matplotlib.patches as patches
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#display the image
plt.imshow(img)
for index in range(len(matchcol)):
    
    y0 = 2048-(matchcol[index]*60) # I think this is the right calculation. I could be wrong.
    x0 = (matchrow[index]*60-index*20)
    plt.gca().add_patch(Rectangle((x0,y0),70,70,        
                    edgecolor='red',
                    facecolor='none',
                    lw=1))


# In[ ]:




