import shutil
import tensorflow as tf
import os
from tflearn.layers.conv import max_pool_2d, conv_2d
from flask import Flask, render_template, request
import cv2  
from tflearn.layers.core import fully_connected, dropout, input_data
import numpy as np  
import tflearn
from tqdm import tqdm  
from tflearn.layers.estimator import regression


app = Flask(__name__)
@app.route('/')
def index():
    return render_template('imageforgerydetection.html')

@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
        folderPath = "static/images"
        directoryList = os.listdir(folderPath)
        for nameOfFile in directoryList:
            os.remove(folderPath + "/" + nameOfFile)
        nameOfFile=request.form['nameOfFile']
        dst = "static/images"
        filedir = "datasets/test"
        shutil.copy(filedir+"/"+nameOfFile, dst)
        
        verify_dir = 'static/images'
        sizeofimage = 50
        rateoflearn = 1e-3
        nameofmodel = 'image_forgery-{}-{}.model'.format(rateoflearn, '2conv-basic')
            
        def processDataVerification():
            dataVerificationInProgress = []
            for imageData in tqdm(os.listdir(verify_dir)):
                resourcePathLoc = os.path.join(verify_dir, imageData)
                imageData = cv2.imread(resourcePathLoc, cv2.IMREAD_COLOR)
                imageData = cv2.resize(imageData, (sizeofimage, sizeofimage))
                dataVerificationInProgress.append(np.array(imageData))
            dataVerificationInProgress = np.array(dataVerificationInProgress)
            np.save('verify_data.npy', dataVerificationInProgress)
            return dataVerificationInProgress

        verify_data = processDataVerification()
        
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

        neural = tflearn.DNN(neuralNetworkModel, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(nameofmodel)):
            neural.load(nameofmodel)
            print('Loaded successfully!')

        acc=""
        
        for data in verify_data:
            inputImage = data
            data = inputImage.reshape(sizeofimage, sizeofimage, 3)
            outputModelData = neural.predict([data])[0]
            print(outputModelData)
            print('model {}'.format(np.argmax(outputModelData)))

            if np.argmax(outputModelData) == 0:
                classificationLabel = 'Orginal'
                acc="The predicted image is original with a probability of {} % ".format(outputModelData[0]*100)
            elif np.argmax(outputModelData) == 1:
                classificationLabel = 'a'
                acc="The predicted image is Copy-Move with a probability of {} % ".format(outputModelData[1]*100)
            elif np.argmax(outputModelData) == 2:
                classificationLabel = 's'
                acc="The predicted image is Splicing with a probability of {} % ".format(outputModelData[2]*100)
            elif np.argmax(outputModelData) == 3:
                classificationLabel = 't'
                acc="The predicted image is Retouched with a probability of {} % ".format(outputModelData[3]*100)
            
            
            if classificationLabel == 'Orginal':
                status = "Orginal"
               
                
            elif classificationLabel == 'a':
                status = "Copy-Move "
               
                
            elif classificationLabel == 's':
                status = " Splicing "
                
            elif classificationLabel == 't':
                status= 'Retouched'
                
        return render_template('imageforgerydetection.html', status=status, accuracy=acc, ImageDisplay="http://127.0.0.1:5000/static/images/"+nameOfFile)
    return render_template('imageforgerydetection.html')


if __name__ == '__main__':
    app.run(debug=True)
