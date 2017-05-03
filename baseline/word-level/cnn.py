
# coding: utf-8

# In[1]:

for i in range(100):
    fname="C:\\LAWHCA\\chinese-sentiment--analysis-preprocess\\neg\\neg."+str(i)+".txt"
    print(fname)
    with open(fname, "r",errors="ignore") as f:
        z=f.read()
    from Translate_eng import translate_eng
    print(translate_eng(z))


# In[2]:



# In[1]:

neg=[]
for i in range(1000):
    fname="C:\\LAWHCA\\chinese-sentiment--analysis-preprocess\\neg\\neg."+str(i)+".txt"
    print(fname)
    with open(fname, "r",errors="ignore") as f:
        neg.append(f.read())


# In[2]:

pos=[]
for i in range(1000):
    fname="C:\\LAWHCA\\chinese-sentiment--analysis-preprocess\\pos\\pos."+str(i)+".txt"
    print(fname)
    with open(fname, "r",errors="ignore") as f:
        pos.append(f.read())


# In[14]:

import random
data_all=pos+neg
data_dict={}
for i in range(1000):
    data_dict[data_all[i]]=1
for i in range(1000):
    data_dict[data_all[i+1000]]=0

print(len(data_dict))
print(data_all[0])
random.shuffle(data_all)
print(data_all[0])
label=[]
for i in range(2000):
    label.append(data_dict[data_all[i]])
    
# put data in label and data_all list




# In[3]:

from Remove_link import remove_link
from Remove_number import remove_number
from Remove_punctuation import remove_punctuation
from Remove_stopwords import remove_stopwords
from Replace_netword import replace_netword
from Replace_repeatwords import replace_repeatwords
from Replace_ywz import replace_ywz
from Translate_eng import translate_eng
import time
for i in range(2000):
    data_all[i]=translate_eng(data_all[i])
    data_all[i]=replace_ywz(data_all[i])
    data_all[i]=replace_repeatwords(data_all[i])
    data_all[i]=replace_netword(data_all[i])
    data_all[i]=remove_stopwords(data_all[i])
    data_all[i]=remove_punctuation(data_all[i])
    data_all[i]=remove_number(data_all[i])
    data_all[i]=remove_link(data_all[i])
    print(i)


# In[6]:

print(data_all[0])
print(data_all[100])
print(data_all[520])


# In[27]:

import gensim
model = gensim.models.Word2Vec.load("C:\\LAWHCA\\word2vec\\word2vec_wx")
print(model.most_similar(u'宾馆'))


# In[28]:

import jieba
all_data=[]
for i in range(2000):
    seg_list=jieba.cut(data_all[i])
    seg_list=" ".join(seg_list)
    all_data.append(seg_list)
print(all_data[0])
print(all_data[1999])
print(type(all_data[0]))
print(data_all[0])
print(data_all[1999])
print(type(data_all[0]))


# In[9]:

MAX=0
res=0
for i in range(2000):
    if MAX< len(all_data[i]):
        res=i
        MAX=len(all_data[i])
print(MAX)
def sentence_to_array(sentence,MAX):
    ret=[]
    import numpy as np
    zero=np.zeros((256))
    for i in sentence:
        try:
            ret.append(model.wv[i])
        except Exception as err:
            ret.append(zero)
    for i in range(MAX-len(sentence)):
        ret.append(zero)
    return ret
res=[]
for i in range(2000):
    res.append(sentence_to_array(all_data[i],MAX))
    print(i)


# In[11]:

import numpy as np
res=np.array(res)


# In[12]:

print(res.shape)


# In[13]:

import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import pickle


X_train,X_test=res[:1500],res[1500:]
y_train,y_test=label[:1500],label[1500:]


# In[3]:

print()
print(X_train.shape)
y_train = np_utils.to_categorical(y_train, 2)
y_test = np_utils.to_categorical(y_test, 2)


# In[4]:

print(X_train.shape)
print(X_test.shape)
X_train = X_train.reshape(1500, 1,1128, 256)
X_test = X_test.reshape(500, 1,1128, 256)


# In[5]:

print(X_train.shape)
print(y_train.shape)



# In[14]:

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
                 1128, 256,)    # height & width
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
model.fit(X_train, y_train, batch_size=50,nb_epoch=11)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)


# In[15]:

data=all_data
print(data[0])
print(data[1])


# In[37]:

for i in range(2000):
    all_data[i]=all_data[i].split()
print(all_data[1])
print(data[1])


# In[38]:

print(all_data[0])


# In[22]:

print(res[0])


# In[39]:

print(all_data[2])


# In[40]:

MAX=0
res=0
for i in range(2000):
    if MAX< len(all_data[i]):
        res=i
        MAX=len(all_data[i])
print(MAX)
def sentence_to_array(sentence,MAX):
    ret=[]
    import numpy as np
    zero=np.zeros((256))
    for i in sentence:
        try:
            ret.append(model.wv[i])
        except Exception as err:
            ret.append(zero)
    for i in range(MAX-len(sentence)):
        ret.append(zero)
    return ret
res=[]
for i in range(2000):
    res.append(sentence_to_array(all_data[i],MAX))
    print(i)


# In[41]:

print(res[0])


# In[33]:

print(all_data[0])


# In[43]:

res=np.array(res)


# In[45]:

import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import pickle


X_train,X_test=res[:1500],res[1500:]
y_train,y_test=label[:1500],label[1500:]


# In[3]:

print()
print(X_train.shape)
y_train = np_utils.to_categorical(y_train, 2)
y_test = np_utils.to_categorical(y_test, 2)


# In[4]:

print(X_train.shape)
print(X_test.shape)
X_train = X_train.reshape(1500, 1,391, 256)
X_test = X_test.reshape(500, 1,391, 256)


# In[5]:

print(X_train.shape)
print(y_train.shape)



# In[46]:

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
                 391, 256,)    # height & width
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
model.fit(X_train, y_train, batch_size=50,nb_epoch=11)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)


# In[ ]:



