import pandas as pd
import numpy as np
from skimage.io import imread
from tqdm import tqdm
import torch

# Load the Trained Model
model = torch.load('./train_learning_base.pt')
# Load Data File
test = pd.read_csv('./dataset/Test/test.csv')
# Load Output File
outXls = pd.read_csv('./result/result_learning.csv')

pathName = []
test_img = []
#loop for store test data
for img_name in tqdm(test['id']):
    image_path = './dataset/Test/' + str(img_name) # defining the image path ex. './dataset/test/1/01ROI.jpg'
    pathName.append(str(img_name)) #add pathName for add in out.csv
    img = imread(image_path, as_gray=True)  # read image to value
    img = img.astype('float32')
    img /= 255.0 #normalize to 0-1
    test_img.append(img)
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


#convert csv to dataframe for change format (auto change row of data in out.csv)
df = pd.DataFrame(data=outXls)
#loop for add row to out.csv if amount of test less than amount of expected output
while(len(test['id']) > len(df['id'])):
    df = df.append({'id': '0','label':'0'}, ignore_index=True)

#loop for delete row in out.csv if amount of test more than amount of expected output
while(len(test['id']) < len(df['id'])):
    df.drop(len(df['id'])-2, axis=0,inplace=True)

# add expected output to out.csv
df['label'] = predictions
df['id'] = pathName
df.head()
# saving the file
df.to_csv('./result/result_learning.csv', index=False)