# -*- coding: utf-8 -*-
"""Experiment2.ipynb

   In this code we have used Relu Activation Function to train the fully connected layers. 
   In this project we don't have used the validation set because of the data scarcity.
"""

#Mounting Google Drive to load the dataset
from google.colab import drive
drive.mount('/content/drive/')

#installing tensorflow-io for decodding the tiff images
!pip install tensorflow-io

#importing packages
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from keras.layers import Input, Lambda, Dense, Flatten , Conv2D , MaxPool2D , AveragePooling2D , Dropout , PReLU , GlobalAveragePooling2D
from keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.initializers import Constant
import glob
import os
import pandas as pd
from tensorflow.keras import Model
import tensorflow_io as tfio
import math
import random

#Defininf image size
IMG_H = 256
IMG_W = 256
IMG_C = 3

#Defining training path
train_path = '/content/drive/MyDrive/WBSUdb_text/'

#Storing handwriting image folder names of every single writter individually
w = os.listdir(train_path)
print("Different No. of Writters: ", len(w))

#Writters and file information
writters = []

for i in w:
 all_writters = os.listdir(train_path+str(i)+'/')
 for c in all_writters:
    writters.append((i, str(train_path+str(i))+'/'+ c))

#Manually splitting train data and test data(80% train and 20% test)
path = train_path
train_images = []
test_images = []
train_labels = []
test_labels = []
for i in w:
  files = os.listdir(train_path+str(i)+'/')
  n = len(files)
  a = (n*(80/100))
  a = math.floor(a)  #train_ids
  b = n-a  #test_ids
  for j in range(0,a):
    k = files[j]
    p = tf.io.read_file(train_path+'/'+str(i)+'/'+k)
    p = tfio.experimental.image.decode_tiff(p)
    p = p[:,:,:IMG_C]
    p = tf.cast(p,tf.float32)
    p = tf.image.resize(p,[IMG_H,IMG_W],method=tf.image.ResizeMethod.AREA)
    p = p/255.0
    train_images.append(p)
    train_labels.append(i)
  for l in range(a,a+b):
    k = files[l]
    p = tf.io.read_file(train_path+'/'+str(i)+'/'+k)
    p = tfio.experimental.image.decode_tiff(p)
    p = p[:,:,:IMG_C]
    p = tf.cast(p,tf.float32)
    p = tf.image.resize(p,[IMG_H,IMG_W],method=tf.image.ResizeMethod.AREA)
    p = p/255.0
    test_images.append(p)
    test_labels.append(i)

#Size of the train and test dataset with corresponding labels
print(len(train_images))
print(len(train_labels))
print(len(test_images))
print(len(test_labels))

plt.imshow(train_images[20])

train_labels[20]

plt.imshow(test_images[100])

test_labels[20]

#Converting the dataset into numpy array
train_images = np.array(train_images)
test_images = np.array(test_images)
print(train_images.shape)
print(test_images.shape)

#Creating one hot encoded labels
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from keras.utils.np_utils import to_categorical
y_labelencoder = LabelEncoder ()
train_y = (y_labelencoder.fit_transform (train_labels))
train_y = to_categorical(train_y,183)
test_y = (y_labelencoder.fit_transform (test_labels))
test_y = to_categorical(test_y,183)

test_y[5]

len(train_y[5])

#Load the model and weights
input = tf.keras.Input(shape=(IMG_H,IMG_W,IMG_C))
vgg = VGG16(include_top=False,weights='imagenet',input_tensor=input)

#VGG16 Model summary(Excluding top layers)
vgg.summary()

#To maintain the accuracy of the new model
for layer in vgg.layers:
  layer.trainable=False

#Mosifying the classification part of the model and adding L1 and L2 Regularization
model = tf.keras.models.Sequential()
model.add(vgg)
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(256,activation='relu',kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5,l2=1e-5),bias_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5,l2=1e-5),activity_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5,l2=1e-5)))
#model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
model.add(Dropout(0.5))
model.add(Dense(256,activation='relu',kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5,l2=1e-5),bias_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5,l2=1e-5),activity_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5,l2=1e-5)))
#model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
model.add(Dropout(0.5))
model.add(Dense(183,activation='softmax'))

model.summary()

#callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)

#Using Adam optimizer
op = tf.keras.optimizers.Adam(learning_rate = 0.001)

#Compiling the model
model.compile(op,loss=tf.keras.losses.categorical_crossentropy,metrics=['accuracy'])

#werertytyuiuiopooppoopppopoopopopopopopppopopopoppopoppppppopopppopoppopooppo

#Training the model and saving the training history
history = model.fit(train_images,train_y,batch_size=100,epochs=300,verbose=1,validation_data=(test_images,test_y))

#Evaluating the model on unseen test dataset
test_loss,test_accuracy = model.evaluate(test_images,test_y)

print("Test Accuracy of the model is: {a:.2f}%".format(a=test_accuracy*100))

#storing the predicted results from the test dataset
predictions = model.predict(test_images)

#Using a threshold of 0.5
#y_pred = (predictions>0.5)

#Confussion matrix
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(test_y.argmax(axis=1),predictions.argmax(axis=1))
print(matrix)

#Training and validation loss curves
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Training and validation accuracy curves
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Random experiment with some randomly taken image data
for i in range(0,200):
  print('True: {} , Predicted: {}'.format(np.argmax(test_y[i]),np.argmax(predictions[i])))

#Saving the model
model.save('Model_Trial_3_No_Validation_ReLU')

!zip -r /content/Model_Trial_3_No_Validation_ReLU.zip /content/Model_Trial_3_No_Validation_ReLU
