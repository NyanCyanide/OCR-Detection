import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.layers
import keras.layers

path1 = "Data\\A-Z"
path2 = "Data\\0-9"

images = []
labels =[]

#Reads The images in Data\\A-Z and append then in images, and label name in Labels
for x in range(0,len(os.listdir(path1))):
    mypiclist = os.listdir(path1 + "//" + str(x))
    for y in mypiclist:
        curimg = cv2.imread(path1 + "//" + str(x) + "//"+ y)
        curimg = cv2.resize(curimg, (32,32))
        images.append(curimg)
        labels.append(x)

#Reads The images in Data\\0-9 and append then in images, and label name in Labels
for x in range(0,len(os.listdir(path2))):
    mypiclist = os.listdir(path2 + "//" + str(x))
    for y in mypiclist:
        curimg = cv2.imread(path2 + "//" + str(x) +"//"+ y)
        curimg = cv2.resize(curimg, (32,32))
        images.append(curimg)
        labels.append(x+26)

images = np.array(images) #Convering list of images into numpy array object
labels = np.array(labels) #Convering list of image labels into numpy array object

#splitting Data into Training and Testing set
X_train,X_test,Y_train,Y_test = train_test_split(images,labels,test_size = 0.2)
X_train,X_validation,Y_train,Y_validation = train_test_split(X_train,Y_train,test_size = 0.2)

numofsamples = []
for x in range(0,26):
    numofsamples.append(len(np.where(Y_train==x)[0]))
for x in range(26,36):
    numofsamples.append(len(np.where(Y_train==x)[0]))

print(numofsamples)

import matplotlib.pyplot as plt   
plt.bar(range(0,36), numofsamples)
plt.title("No of images for each class")
plt.xlabel("CLASS ID")
plt.ylabel("Number of Images")
plt.show()

def preprocessing(img):
    """Converts colored image into gray scale, makes all the intensities equal and making"""
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

#passing an image, to be converted into grayscale image and Histogram Equalization
X_train = np.array(list(map(preprocessing,X_train)))
X_test = np.array(list(map(preprocessing,X_test)))
X_validation = np.array(list(map(preprocessing,X_validation)))

#Reshaping the size, so that it recognizes as grayscal image
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validation = X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)


#Apply random transformations to each image in the batch. Replacing the original batch of images with a new randomly transformed batch.
datagen = ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,zoom_range=0.2,shear_range=0.1,rotation_range=10)
datagen.fit(X_train)


Y_train = to_categorical(Y_train,36)
Y_test = to_categorical(Y_test,36)
Y_validation = to_categorical(Y_validation,36)



def mymodel():
    nooffilters = 60
    sizeoffilter1 = (5,5)
    sizeoffilters2 = (3,3)
    sizeofPool = (2,2)
    noofnode = 500

    model = Sequential()
    model.add((Conv2D(nooffilters,sizeoffilter1,input_shape =(32,32,1),activation = "relu")))
    model.add((Conv2D(nooffilters,sizeoffilter1,activation = "relu")))
    model.add(MaxPooling2D(pool_size=sizeofPool))
    model.add(Conv2D(nooffilters//2,sizeoffilters2,activation="relu"))
    model.add(Conv2D(nooffilters//2,sizeoffilters2,activation="relu"))
    model.add(MaxPooling2D(pool_size=sizeofPool))
    model.add(tensorflow.keras.layers.Dropout(0.5))
    model.add(tensorflow.keras.layers.Flatten())
    model.add(keras.layers.Dense(noofnode,activation = "relu"))
    model.add(tensorflow.keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(36,activation = "softmax"))
    model.compile(Adam(lr = 0.001),loss="categorical_crossentropy",metrics = ["accuracy"])
    return model

model = mymodel()
print(model.summary())

batchsizeval = 50 #setting batch size to 50
epochsval = 50 #setting No of iteration to 50
stepsperepoch = len(X_train)//batchsizeval #No of batches to be done

#Trains the data
history =  model.fit_generator(datagen.flow(X_train,Y_train,batch_size=batchsizeval),steps_per_epoch=stepsperepoch,epochs = epochsval,validation_data=(X_validation,Y_validation),shuffle = 1)

#plot for accuracy and loss
plt.figure(1)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["training","validation"])
plt.title("Loss")
plt.xlabel("Epochs")
    
plt.figure(2)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.legend(["training","validation"])
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.show()

score = model.evaluate(X_test,Y_test,verbose = 0)
print("Test Score : ",score[0]) # more the score the better
print("Test Accuracy : ",score[1]) # more the accuracy, best accuaracy a model can give

#trained model is saved here
model.save("Model1.h5")

