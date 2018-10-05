
# coding: utf-8

# In[1]:


import numpy as np
from keras import layers
from keras.models import Sequential
import pandas as pd
from keras.utils import np_utils,print_summary
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from keras.layers import Input,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D
from keras.layers import AveragePooling2D,MaxPooling2D,Dropout,GlobalMaxPooling2D,GlobalAveragePooling2D


# In[2]:


data=pd.read_csv("C:\\Users\\Sushant\\EMOJINATOR\\gestures\\train_foo1.csv")
dataset=np.array(data)
Y=dataset[:,0]
print(Y)
np.random.shuffle(dataset)
X=dataset[:,1:2501]
Y=dataset[:,0]
Y.shape


# In[3]:


Y = Y.reshape(Y.shape[0], 1)
X_train=X[0:12001,:]
X_train=X_train/255.
X_test=X[12001:,:]
X_test=X_test/255

Y_train=Y[0:12001,:]
Y_train=Y_train.T
Y_test=Y[12001:,:]
Y_test=Y_test.T
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[4]:


image_x=50
image_y=50
train_y=np_utils.to_categorical(Y_train)
test_y=np_utils.to_categorical(Y_test)
train_y = np_utils.to_categorical(Y_train)
test_y = np_utils.to_categorical(Y_test)
train_y = train_y.reshape(train_y.shape[1], train_y.shape[2])
test_y = test_y.reshape(test_y.shape[1], test_y.shape[2])
X_train = X_train.reshape(X_train.shape[0], 50, 50, 1)
X_test = X_test.reshape(X_test.shape[0], 50, 50, 1)
print(train_y.shape)
print(test_y.shape)
print(X_test.shape)
print(X_train.shape)


# In[5]:


def keras_model(image_x,image_y):
    no_of_classes=10
    model=Sequential()
    model.add(Conv2D(filters=32,kernel_size=(5,5),input_shape=(image_x,image_y,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))

    model.add(Conv2D(filters=64,kernel_size=(5,5),activation='relu'))
    model.add(MaxPooling2D(pool_size=(5,5),strides=(5,5),padding='same'))
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(no_of_classes,activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath="face_rec1.h5"
    checkpoint1=ModelCheckpoint (filepath, monitor='val_acc', verbose=1, save_best_only=True, mode="max")
    callbacks_list=[checkpoint1]
    return model,callbacks_list


# In[6]:


model,callbacks_list=keras_model(image_x,image_y)
model.fit(X_train,train_y,validation_data=(X_test,test_y),epochs=2,batch_size=64,callbacks=callbacks_list)
scores=model.evaluate(X_test,test_y,verbose=0)
print("cnn error:%0.2f%%"% (100-scores[1]*100))
print_summary(model)
model.save('face_rec1.h5')

