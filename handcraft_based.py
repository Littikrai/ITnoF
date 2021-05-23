import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import sklearn.neighbors as sn
import tqdm as t
import numpy as np
import pandas as pd

featureTr = []
labelTr = []

# Training Image

for _classname in t.tqdm(range(1, 11)):
    for id in range(1, 11):
        path = ('./dataset/Train/'+str(_classname)+'/0'+ str(id)+'ROI.jpg')
        image = imread(path)
        #resize Image
        re_image = resize(image, (64, 128))
        # Feature Extraction
        fd = hog(re_image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        featureTr.append(fd)
        labelTr.append(_classname)

featureTr = np.reshape(np.array(featureTr), (100, -1))


# Testing Image
test = pd.read_csv('./dataset/test/test.csv')
outXls = pd.read_csv('./result/result_handcraft.csv')
test_img = []
pathName = []
predictions = []
# for test image
for img_name in t.tqdm(test['id']):
    pathT = './dataset/Test/'+str(img_name)
    imgT = imread(pathT)
    pathName.append(str(img_name))
    re_imageT = resize(imgT, (64, 128))
    # Feature Extraction
    featureTs = hog(re_imageT, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    featureTs = featureTs.reshape(1, -1)
    test_img.append(featureTs)

# k-nn Classification
classifier = sn.KNeighborsClassifier(n_neighbors=1)
classifier.fit(featureTr, labelTr)

for img in range(len(test_img)):
    out = classifier.predict(test_img[img])
    predictions.append(int(out))

df = pd.DataFrame(data=outXls)
while(len(test['id']) > len(df['id'])):
    df = df.append({'id': '0','label':'0'}, ignore_index=True)

while(len(test['id']) < len(df['id'])):
    df.drop(len(df['id'])-2, axis=0,inplace=True)

df['label'] = predictions
df['id'] = pathName
df.head()
# saving the file
df.to_csv('./result/result_handcraft.csv', index=False)





