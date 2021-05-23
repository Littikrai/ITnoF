import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pandas as pd
import numpy as np
from Net import Net
# for reading and displaying images
from skimage.io import imread
# for evaluating the model
from tqdm import tqdm
# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

train = pd.read_csv('./dataset/train/train.csv')
train.head()
train_img = []
for img_name in tqdm(train['id']):
        # defining the image path
        impath = ('./dataset/train/'+str(img_name))
        # reading the image
        img = imread(impath, as_gray=True)
        # # converting the type of pixel to float
        img = img.astype('float32')
        img /= 255.0
        # appending the image into the list
        train_img.append(img)
        # converting the list to numpy array
train_x = np.array(train_img)
# defining the target
train_y = train['label'].values

# converting training images into torch format
train_x = train_x.reshape(100, 1, 110, 220)
train_x = torch.from_numpy(train_x).to(torch.float32)
# converting the target into torch format
train_y = train_y.astype(int)
train_y = torch.from_numpy(train_y).to(torch.float32)

# defining the model
model = Net()
# defining the optimizer
optimizer = SGD(model.parameters(), lr=0.03, momentum=0.9)
# defining the loss function
criterion = CrossEntropyLoss()

#empty list to store training losses
train_losses = []
# empty list to store validation losses
val_losses = []
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

    # computing the training and validation loss
    # we convert the results because they aren't in the good format
    y_train = y_train.long() 
    y_train = y_train.squeeze_()
    loss_train = criterion(output_train, y_train)
    train_losses.append(loss_train)
    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()

# Saving Entire Model 
# A common PyTorch convention is to save models using either a .pt or .pth file extension.
torch.save(model, './model.pt')