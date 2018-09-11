from __future__ import print_function
import numpy as np
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.datasets import fashion_mnist
from keras.models import Model
from keras.layers import Input, Dense
from keras.models import load_model
import cv2
import os
import glob
from os import path
import random

#input image dimensions:
img_rows=80
img_cols=80
#number of channels
img_channels=3

save_path='./save_images'
def label(img):
#    tag=img.split('.')[0]
#    return tag
#
    label = img.split('.')[0]
    if label== 'traf':
        ohl= np.array([1,0])
        print (label)
    else:
        print(label)
        ohl = np.array([0,1])
    return ohl

#def create_image_lists():
mypath='./trainthis'
testpath='./testthis'
x_train = []
y_train = []
x_test = []
y_test = []
file_list=os.listdir(mypath)
num_samples=np.size(file_list)
#for n in range(0, len(onlyfiles)):
#    x_train.append(cv2.imread( path.join(mypath,onlyfiles[n]) ))
#    #convert img to grayscale if needed
#    #x_train=x_train.convert('L')
#    #save the processed imges
#    #x_train.save(save_path+'/'+ n,".jpg")
#    #read label from folder
#    y_train.append()
def get_testdata():
    for n in os.listdir(testpath):
        path=os.path.join(testpath,n)
        img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img, (img_rows,img_cols))
        x_test.append(np.array(img))
        y_test.append(label(n))

img_num=0
def get_traindata():
    img_num=0
    for n in os.listdir(mypath):
        try:
            img_num+=1
            path=os.path.join(mypath,n)
            img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img=cv2.resize(img, (img_rows,img_cols))
            x_train.append(np.array(img))
            y_train.append(label(n))
        except Exception as e:
            print(str(e))
#batch_size
batch_size=15
#number of classes
num_classes=2
#number of epochs to train
epochs=15
#input_shape
input_shape= (img_rows,img_cols,1)
#nb of filters
nb_filters=32
#pooling size for Maxpool
nb_pool=2
#kernel size
kernel_size=3
get_traindata()
get_testdata()
#x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#shuffle the data
c = list(zip(x_train,y_train))
random.shuffle(c)
x_train, y_train=zip(*c)

x_train=np.array(x_train)
x_test=np.array(x_test)
x_train=x_train.reshape(len(x_train),img_rows,img_cols,1)
x_test=x_test.reshape(len(x_test),img_rows,img_cols,1)

#image = np.expand_dims(image, axis=0))
model=Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32,kernel_size=3,strides=1,padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=5,padding='same'))
model.add(Conv2D(filters=32,kernel_size=3,strides=1,padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=5,padding='same'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, np.array(y_train),
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(np.array(x_test),np.array(y_test)))
score = model.evaluate(np.array(x_test),np.array(y_test), verbose=0)
model.save('traffic_detection.model')
new_model=model.load_model('traffic_detection.model')
print('Test loss:', score[0])
print('Test accuracy:', score[1])

