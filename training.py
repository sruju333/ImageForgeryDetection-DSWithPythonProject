import numpy as np         
import os                  
from tqdm import tqdm
from random import shuffle 
from tensorflow.python.framework import ops
import cv2


folderWithTrainData = 'datasets\\train'
folderWithTestData = 'datasets\\test'

sizeofimage = 50
rateoflearn = 1e-3
nameofmodel = 'image_forgery-{}-{}.model'.format(rateoflearn, '2conv-basic')


def imageAssignLabel(imageForgeryPic):
    labelForWord = imageForgeryPic[0]
    print(labelForWord)
  
    if labelForWord == 'O':
        print('Orginal')
        return [1,0,0,0]
    
    elif labelForWord == 'A':
        print('a')
        return [0,1,0,0]
    elif labelForWord == 'S':
        print('s')
        return [0,0,1,0]
    elif labelForWord == 'T':
        print('t')
        return [0,0,0,1]

def trainingDataCreate():
    dataUsedForTraining = []
    for imageForgeryPic in tqdm(os.listdir(folderWithTrainData)):
        classification = imageAssignLabel(imageForgeryPic)
        print('***********')
        print(classification)
        resourcePathLoc = os.path.join(folderWithTrainData,imageForgeryPic)
        imageForgeryPic = cv2.imread(resourcePathLoc,cv2.IMREAD_COLOR)
        imageForgeryPic = cv2.resize(imageForgeryPic, (sizeofimage,sizeofimage))
       
        dataUsedForTraining.append([np.array(imageForgeryPic),np.array(classification)])
    shuffle(dataUsedForTraining)
    np.save('train_data.npy', dataUsedForTraining)
    return dataUsedForTraining

def testDataProcessing():
    dataUsedForTesting = []
    for imageForgeryPic in tqdm(os.listdir(folderWithTestData)):
        resourcePathLoc = os.path.join(folderWithTestData,imageForgeryPic)
        numberOfImage = imageForgeryPic.split('.')[0]
        imageForgeryPic = cv2.imread(resourcePathLoc,cv2.IMREAD_COLOR)
        imageForgeryPic = cv2.resize(imageForgeryPic, (sizeofimage,sizeofimage))
        dataUsedForTesting.append([np.array(imageForgeryPic), numberOfImage])
        
    shuffle(dataUsedForTesting)
    np.save('test_data.npy', dataUsedForTesting)
    return dataUsedForTesting

dataForTrain = trainingDataCreate()


import tflearn
from tflearn.layers.conv import max_pool_2d, conv_2d
from tflearn.layers.core import fully_connected, dropout, input_data
from tensorflow.python.framework import ops
from tflearn.layers.estimator import regression

ops.reset_default_graph()

neuralNetworkModel = input_data(shape=[None, sizeofimage, sizeofimage, 3], name='input')

neuralNetworkModel = conv_2d(neuralNetworkModel, 32, 3, activation='relu')
neuralNetworkModel = max_pool_2d(neuralNetworkModel, 3)

neuralNetworkModel = conv_2d(neuralNetworkModel, 64, 3, activation='relu')
neuralNetworkModel = max_pool_2d(neuralNetworkModel, 3)

neuralNetworkModel = conv_2d(neuralNetworkModel, 128, 3, activation='relu')
neuralNetworkModel = max_pool_2d(neuralNetworkModel, 3)

neuralNetworkModel = conv_2d(neuralNetworkModel, 32, 3, activation='relu')
neuralNetworkModel = max_pool_2d(neuralNetworkModel, 3)

neuralNetworkModel = conv_2d(neuralNetworkModel, 64, 3, activation='relu')
neuralNetworkModel = max_pool_2d(neuralNetworkModel, 3)

neuralNetworkModel = fully_connected(neuralNetworkModel, 2048, activation='relu')
neuralNetworkModel = dropout(neuralNetworkModel, 0.8)

neuralNetworkModel = fully_connected(neuralNetworkModel, 4, activation='softmax')
neuralNetworkModel = regression(neuralNetworkModel, optimizer='adam', learning_rate=rateoflearn, loss='categorical_crossentropy', name='targets')

neural = tflearn.DNN(neuralNetworkModel, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(nameofmodel)):
    neural.load(nameofmodel)
    print('Loaded Successfully!')

dataTRN = dataForTrain[:-1]
dataTST = dataForTrain[-30:]

splitA = np.array([i[0] for i in dataTRN]).reshape(-1,sizeofimage,sizeofimage,3)
splitB = [i[1] for i in dataTRN]
print(splitA.shape)
ASplitTest = np.array([i[0] for i in dataTST]).reshape(-1,sizeofimage,sizeofimage,3)
BSplitTest = [i[1] for i in dataTST]
print(ASplitTest.shape)

modelFitting = neural.fit({'input': splitA}, {'targets': splitB},n_epoch=100, validation_set=({'input': ASplitTest}, {'targets': BSplitTest}),snapshot_step=30, show_metric=True, run_id=nameofmodel)

neural.save(nameofmodel)