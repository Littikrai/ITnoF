import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import sklearn.neighbors as sn
import tqdm as t
import numpy as np

featureTr = []
labelTr = []

# Training Image
for _classname in t.tqdm(range(1, 11)):
    for id in range(1, 11):
        path = ('./dataset/Train/'+str(_classname)+'/0'+ str(id)+'ROI.jpg')
        image = imread(path)
        re_image = resize(image, (64, 128))
        fd = hog(re_image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        featureTr.append(fd)
        labelTr.append(_classname)

featureTr = np.reshape(np.array(featureTr), (100, -1))


# Testing Image
# pathT = './dataset/Train/10/01ROI.jpg'
pathT = './dataset/88.jpg'
imgT = imread(pathT)
re_imageT = resize(imgT, (64, 128))
featureTs = hog(re_imageT, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
featureTs = featureTs.reshape(1, -1)

# # k-nn Classification
classifier = sn.KNeighborsClassifier(n_neighbors=1)
classifier.fit(featureTr, labelTr)
out = classifier.predict(featureTs)
print(out)



