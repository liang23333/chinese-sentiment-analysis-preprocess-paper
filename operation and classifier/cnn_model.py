
# coding: utf-8

# In[2]:

import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import pickle

with open("C:\\LAWHCA\\chinese-sentiment--analysis-preprocess\\label.pkl",'rb') as f:
    label=pickle.load(f)

print(len(label))
print(label)
with open("C:\\LAWHCA\\chinese-sentiment--analysis-preprocess\\wordvec.pkl",'rb') as f:
    wordvec=pickle.load(f)

print(type(wordvec))
print(wordvec[0])
wordvec=np.array(wordvec)
X_train,X_test=wordvec[:1500],wordvec[1500:]
y_train,y_test=label[:1500],label[1500:]


# In[3]:

print()
print(X_train.shape)
y_train = np_utils.to_categorical(y_train, 2)
y_test = np_utils.to_categorical(y_test, 2)


# In[4]:

print(X_train.shape)
print(X_test.shape)
X_train = X_train.reshape(1500, 1,2033, 256)
X_test = X_test.reshape(500, 1,2033, 256)


# In[5]:

print(X_train.shape)
print(y_train.shape)


# In[ ]:

# Another way to build your CNN
model = Sequential()

# Conv layer 1 output shape (32, 28, 28)
model.add(Convolution2D(
    nb_filter=32,
    nb_row=5,
    nb_col=5,
    border_mode='same',     # Padding method
    dim_ordering='tf',      # if use tensorflow, to set the input dimension order to theano ("th") style, but you can change it.
    input_shape=(1,         # channels
                 2033, 256,)    # height & width
))
model.add(Activation('relu'))

# Pooling layer 1 (max pooling) output shape (32, 14, 14)
model.add(MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2),
    border_mode='same',    # Padding method
))

# Conv layer 2 output shape (64, 14, 14)
model.add(Convolution2D(64, 5, 5, border_mode='same'))
model.add(Activation('relu'))

# Pooling layer 2 (max pooling) output shape (64, 7, 7)
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))

model.add(Convolution2D(128,5,5, border_mode='same'))
model.add(Activation('relu'))

# Pooling layer 2 (max pooling) output shape (64, 7, 7)
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))

# Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

# Fully connected layer 2 to shape (10) for 10 classes
model.add(Dense(2))
model.add(Activation('softmax'))

# Another way to define your optimizer
adam = Adam(lr=1e-4)

# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, batch_size=100,nb_epoch=11)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)


# In[6]:




# In[13]:




# In[8]:




# In[14]:




# In[22]:




# In[24]:




# In[25]:




# In[28]:




# In[29]:




# In[30]:




# In[38]:




# In[ ]:



