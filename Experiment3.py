# -*- coding: utf-8 -*-
"""Experiment3.ipynb


   In this code we have used LeakyRelu Activation Function to train the fully connected     layers.In this project we have used the validation set to validate the model.

"""

#Mounting Google Drive to load the dataset
from google.colab import drive
drive.mount('/content/drive/')

!pip install tensorflow-io

#importing packages
import os
import cv2
import glob
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio
from keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from keras.models import Sequential
from keras.preprocessing import image
from keras.initializers import Constant
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from tensorflow.keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Input, Lambda, Dense, Flatten , Conv2D , MaxPool2D , AveragePooling2D , Dropout , PReLU

#Defininf image size
IMG_H = 256
IMG_W = 256
IMG_C = 3

#Defining training path
train_path = '/content/drive/MyDrive/WBSUdb_text/'

#Storing handwriting image folder names of every single writter individually
w = os.listdir(train_path)

#Manually splitting train data and test data(80% train and 20% test)
path = train_path
train_images = []
test_images = []
val_images = []
train_labels = []
test_labels = []
val_labels = []
count = 0
for i in w:
  files = os.listdir(train_path+str(i)+'/')
  n = len(files)
  if (n!=2):
    a = (n*(60/100))
    a = math.floor(a)  #train_ids
    b = n-a  
    z = (b*(50/100))
    z = math.floor(z) #test_ids
    h = b-z #val_ids
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
    for l in range(a,a+z):
      k = files[l]
      p = tf.io.read_file(train_path+'/'+str(i)+'/'+k)
      p = tfio.experimental.image.decode_tiff(p)
      p = p[:,:,:IMG_C]
      p = tf.cast(p,tf.float32)
      p = tf.image.resize(p,[IMG_H,IMG_W],method=tf.image.ResizeMethod.AREA)
      p = p/255.0
      test_images.append(p)
      test_labels.append(i)
    for u in range(a+z,a+z+h):
      k = files[u]
      p = tf.io.read_file(train_path+'/'+str(i)+'/'+k)
      p = tfio.experimental.image.decode_tiff(p)
      p = p[:,:,:IMG_C]
      p = tf.cast(p,tf.float32)
      p = tf.image.resize(p,[IMG_H,IMG_W],method=tf.image.ResizeMethod.AREA)
      p = p/255.0
      val_images.append(p)
      val_labels.append(i)
    count = count+1

print("Different No. of Writters: {}".format(count))

#Size of the train and test dataset with corresponding labels
print(len(train_images))
print(len(train_labels))
print(len(test_images))
print(len(test_labels))
print(len(val_images))
print(len(val_labels))

plt.imshow(train_images[10])

plt.imshow(test_images[10])

plt.imshow(val_images[10])

train_labels[10]

test_labels[10]

val_labels[10]

#Converting the dataset into numpy array
train_images = np.array(train_images)
test_images = np.array(test_images)
val_images = np.array(val_images)
print(train_images.shape)
print(test_images.shape)
print(val_images.shape)

#Creating one hot encoded labels
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from keras.utils.np_utils import to_categorical
y_labelencoder = LabelEncoder ()
train_y = (y_labelencoder.fit_transform (train_labels))
train_y = to_categorical(train_y,count)
test_y = (y_labelencoder.fit_transform (test_labels))
test_y = to_categorical(test_y,count)
val_y = (y_labelencoder.fit_transform (val_labels))
val_y = to_categorical(val_y,count)

print(len(train_y),len(test_y),len(val_y))

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
model.add(Dense(256,kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5,l2=1e-5),bias_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5,l2=1e-5),activity_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5,l2=1e-5)))
model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
model.add(Dropout(0.5))
model.add(Dense(256,kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5,l2=1e-5),bias_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5,l2=1e-5),activity_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5,l2=1e-5)))
model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
model.add(Dropout(0.5))
model.add(Dense(count,activation='softmax'))

model.summary()

#2666

#Using Adam optimizer
op = tf.keras.optimizers.Adam(learning_rate = 0.001)

#Compiling the model
model.compile(op,loss=tf.keras.losses.categorical_crossentropy,metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

#Training the model and saving the training historyiooioioioio
history = model.fit(train_images,train_y,batch_size=100,epochs=300,verbose=1,validation_data=(val_images,val_y),callbacks=callback)

#Evaluating the model on unseen test dataset
test_loss,test_accuracy = model.evaluate(test_images,test_y)

#Evaluating validation loss
val_loss,val_accuracy = model.evaluate(val_images,val_y)

print("Test Accuracy of the model is: {a:.2f}%".format(a=test_accuracy*100))

print("Validation Accuracy of the model is: {a:.2f}%".format(a=val_accuracy*100))

predictions = model.predict(test_images)

#Confussion matrix
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(test_y.argmax(axis=1),predictions.argmax(axis=1))
print(matrix)



#Random experiment with some randomly taken image data
for i in range(0,200):
  print('True: {} , Predicted: {}'.format(np.argmax(test_y[i]),np.argmax(predictions[i])))

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

model.save('Trial_Model')

model.save_weights('TRial_Model_Weights')

!zip -r /content/Trial_Model.zip /content/Trial_Model
