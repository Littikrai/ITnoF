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
train_x = np.array(train_img)
train_y = train['label'].values # expected output value

# converting training images into torch format
train_x = train_x.reshape(100, 1, 110, 220)
train_x = torch.from_numpy(train_x).to(torch.float32)
# converting the target into torch format
train_y = train_y.astype(int)
train_y = torch.from_numpy(train_y).to(torch.float32)

# defining the model
model = FKPStructure()

# defining the optimizer
optimizer = SGD(model.parameters(), lr=0.03, momentum=0.9)

# defining the loss function
criterion = CrossEntropyLoss()

#empty list to store training losses
train_losses = []
# defining the number of epochs
n_epochs = 25
# training the model
for epoch in tqdm(range(n_epochs)):
    model.train()
    tr_loss = 0
    # getting the training set
    x_train, y_train = Variable(train_x), Variable(train_y)

    # clearing the Gradients of the model parameters
    optimizer.zero_grad()
    # prediction for training and validation set
    output_train = model(x_train)

    y_train = y_train.long() 
    y_train = y_train.squeeze_()
    loss_train = criterion(output_train, y_train)
    train_losses.append(loss_train)
    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()

# Saving Model to this path
torch.save(model, './FKPStructure.pt')
