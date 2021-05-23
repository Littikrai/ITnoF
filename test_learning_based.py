import pandas as pd
import numpy as np
# for reading and displaying images
from skimage.io import imread
# for evaluating the model
from tqdm import tqdm
# PyTorch libraries and modules
import torch

# Load the Trained Model from model04.pt
model = torch.load('./model.pt')

# Load Data File
test = pd.read_csv('./dataset/test/test.csv')
outXls = pd.read_csv('./result/result_learning.csv')
pathName = []
# loading test images
test_img = []
for img_name in tqdm(test['id']):
    # defining the image path
    image_path = './dataset/test/' + str(img_name)
    pathName.append(str(img_name))
    # reading the image
    img = imread(image_path, as_gray=True)
    # normalizing the pixel values
    img = np.divide(img, 255.0)
    # converting the type of pixel to float 32
    img = img.astype('float32')
    # appending the image into the list
    test_img.append(img)
# converting the list to numpy array
test_x = np.array(test_img)
# converting training images into torch format
test_x = test_x.reshape(len(test['id']), 1, 110, 220)
test_x = torch.from_numpy(test_x).to(torch.float32)

# generating predictions for test set
with torch.no_grad():
    output = model(test_x)

softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)
print(predictions)

df = pd.DataFrame(data=outXls)
while(len(test['id']) > len(df['id'])):
    df = df.append({'id': '0','label':'0'}, ignore_index=True)

while(len(test['id']) < len(df['id'])):
    df.drop(len(df['id'])-2, axis=0,inplace=True)


# replacing the label with prediction
df['label'] = predictions
df['id'] = pathName
df.head()
# saving the file
df.to_csv('./result/result_learning.csv', index=False)