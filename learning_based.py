import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pandas as pd
import numpy as np
from FKPStructure import FKPStructure
from skimage.io import imread
from tqdm import tqdm
import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

train = pd.read_csv('./dataset/Train/train.csv')
train.head()
train_img = []
#loop for store train data
for img_name in tqdm(train['id']):
        impath = ('./dataset/Train/'+str(img_name)) # defining the image path ex. './dataset/train/1/01ROI.jpg'
        img = imread(impath, as_gray=True) # read image to value
        img = img.astype('float32')
        img /= 255.0 #normalize to 0-1
        train_img.append(img)
train_data = np.array(train_img) # trainning data
train_edata = train['label'].values # expected data

# converting into torch format
train_data = train_data.reshape(100, 1, 110, 220)
train_data = torch.from_numpy(train_data).to(torch.float32)
train_edata = train_edata.astype(int)
train_edata = torch.from_numpy(train_edata).to(torch.float32)

# defining the model, optimizer, loss function
model = FKPStructure()
optimizer = SGD(model.parameters(), lr=0.03, momentum=0.9)
criterion = CrossEntropyLoss()

train_losses = [] #list for store training losses
n_epochs = 25 # defining the number of epochs

# training the model
for epoch in tqdm(range(n_epochs)):
    model.train()
    tr_loss = 0
    data, edata = Variable(train_data), Variable(train_edata)  # getting the training set
    optimizer.zero_grad()  # clearing the Gradients of the model parameters
    output_train = model(data)

    edata = edata.long()
    edata = edata.squeeze_()
    loss_train = criterion(output_train, edata)
    train_losses.append(loss_train)
    
    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()

# Saving Model
torch.save(model, './train_learning_base.pt')
