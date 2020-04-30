import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DATADIR = "D:/TrainingData/PetImages"
CATEGORIES = ["Dog", "Cat"]


IMG_SIZE = 50
trainingData = []


def createTrainingData():

    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                trainingData.append([new_array, class_num])
            except Exception as e:

                pass


createTrainingData()
print(len(trainingData))

random.shuffle(trainingData)

for sample in trainingData[:10]:
    print(sample[1])


X = []
Y = []


for feature, label in trainingData:
    X.append(feature)
    Y.append(label)


X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = np.array(Y).reshape(-1, 1)


pickleOut = open("datasets/X.pickle","wb")
pickle.dump(X, pickleOut)
pickleOut.close()

pickleOut = open("datasets/Y.pickle","wb")
pickle.dump(Y, pickleOut)
pickleOut.close()
'''
to read:
    pickleIn = open("X.pickle", "rb")
    pickle.load(pickleIn)

'''
