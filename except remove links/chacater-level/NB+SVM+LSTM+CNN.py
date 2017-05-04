
# coding: utf-8

# In[1]:



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




# In[2]:

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
    #data_all[i]=remove_link(data_all[i])
    print(i)


# In[3]:

print(data_all[100])


# In[4]:

from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
tfidf = TFIDF(min_df=5, # 最小支持度为2
           max_features=None,
           strip_accents='unicode',
           analyzer='char',
           ngram_range=(1,1),  # 1元文法模型
           use_idf=1,
           smooth_idf=1,
           sublinear_tf=1)


# In[5]:

all_data=data_all
tfidf.fit(all_data)
all_data = tfidf.transform(all_data)
print(type(all_data))


# In[6]:

print(tfidf.vocabulary_)


# In[7]:

print(all_data.shape)


# In[8]:

from sklearn.naive_bayes import MultinomialNB as MNB

model_NB = MNB()
model_NB.fit(all_data[:1500], label[:1500])
MNB(alpha=1.0, class_prior=None, fit_prior=True)

from sklearn.cross_validation import cross_val_score
import numpy as np

print("多项式贝叶斯分类器10折交叉验证得分: ", np.mean(cross_val_score(model_NB, all_data[:1500], label[:1500], cv=10, scoring='roc_auc')))
sum=0
test_predicted =model_NB.predict(all_data[1500:])
for i in range(500):
    if(test_predicted[i]==label[1500+i]):
        sum=sum+1
print(sum/500)


# In[9]:

for i in range(20,21):
    import numpy as np
    from sklearn.decomposition import PCA   
    pca=PCA(n_components=i)  
    newData=pca.fit_transform(all_data.toarray())

    from sklearn import svm
    CLF=svm.SVC()
    CLF.fit(newData[:1500],label[:1500])
    from sklearn.cross_validation import cross_val_score

    print("SVM分类器10折交叉验证得分: ", np.mean(cross_val_score(CLF, newData[:1500], label[:1500], cv=10, scoring='roc_auc')))
    sum=0
    test_predicted =CLF.predict(newData[1500:])
    for i in range(500):
        if(test_predicted[i]==label[1500+i]):
            sum=sum+1
    print(sum/500)


# In[10]:


from sklearn.decomposition import PCA
print(all_data.shape)
pca=PCA(n_components=20)
data=pca.fit_transform(all_data.toarray())
print(data.shape)
max_sequence_length=data.shape[1]


import os
import re
import io
import requests
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from zipfile import ZipFile
from tensorflow.python.framework import ops
ops.reset_default_graph()

sess = tf.Session()

# Set RNN parameters
epochs = 500
batch_size = 100

rnn_size = 10
embedding_size = 1
min_word_frequency = 10
learning_rate = 0.0005
dropout_keep_prob = tf.placeholder(tf.float32)

data=data.reshape((2000,max_sequence_length,1))
label=np.array(label)



x_train, x_test = data[:1500], data[1500:]
y_train, y_test = label[:1500], label[1500:]


x_data = tf.placeholder(tf.float32, [None, max_sequence_length,embedding_size])
y_output = tf.placeholder(tf.int32, [None])

if tf.__version__[0]>='1':
    cell=tf.contrib.rnn.BasicRNNCell(num_units = rnn_size)
else:
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units = rnn_size)

output, state = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
output = tf.nn.dropout(output, dropout_keep_prob)

# Get output of RNN sequence
output = tf.transpose(output, [1, 0, 2])
last = tf.gather(output, int(output.get_shape()[0]) - 1)


weight = tf.Variable(tf.truncated_normal([rnn_size, 2], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape=[2]))
logits_out = tf.nn.softmax(tf.matmul(last, weight) + bias)

# Loss function
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_out, labels=y_output) # logits=float32, labels=int32
loss = tf.reduce_mean(losses)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_out, 1), tf.cast(y_output, tf.int64)), tf.float32))

optimizer = tf.train.RMSPropOptimizer(learning_rate)
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []
# Start training
for epoch in range(epochs):

    # Shuffle training data
    shuffled_ix = np.random.permutation(np.arange(len(x_train)))
    # print(shuffled_ix)
    # print(shuffled_ix.shape[0])
    x_train = x_train[shuffled_ix]
    y_train = y_train[shuffled_ix]
    num_batches = int(len(x_train)/batch_size)
    # TO DO CALCULATE GENERATIONS ExACTLY
    for i in range(num_batches):
        # Select train data
        min_ix = i * batch_size
        max_ix = np.min([len(x_train), ((i+1) * batch_size)])
        x_train_batch = x_train[min_ix:max_ix]
        y_train_batch = y_train[min_ix:max_ix]
        #print(x_train_batch)
        # print(x_train_batch.shape[0])
        # print(x_train_batch.shape[1])
        # print(x_train_batch.shape[2])

        # print(x_train_batch[0][0])
        # Run train step
        train_dict = {x_data: x_train_batch, y_output: y_train_batch, dropout_keep_prob:0.5}
        sess.run(train_step, feed_dict=train_dict)
        
    # Run loss and accuracy for training
    temp_train_loss, temp_train_acc = sess.run([loss, accuracy], feed_dict=train_dict)
    train_loss.append(temp_train_loss)
    train_accuracy.append(temp_train_acc)
    
    # Run Eval Step
    test_dict = {x_data: x_test, y_output: y_test, dropout_keep_prob:1.0}
    temp_test_loss, temp_test_acc = sess.run([loss, accuracy], feed_dict=test_dict)
    test_loss.append(temp_test_loss)
    test_accuracy.append(temp_test_acc)
    print('Epoch: {}, Test Loss: {:.2}, Test Acc: {:.2}'.format(epoch+1, temp_test_loss, temp_test_acc))



# epoch_seq = np.arange(1, epochs+1)
# plt.plot(epoch_seq, train_loss, 'k--', label='Train Set')
# plt.plot(epoch_seq, test_loss, 'r-', label='Test Set')
# plt.title('Softmax Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Softmax Loss')
# plt.legend(loc='upper left')
# plt.show()

# # Plot accuracy over time
# plt.plot(epoch_seq, train_accuracy, 'k--', label='Train Set')
# plt.plot(epoch_seq, test_accuracy, 'r-', label='Test Set')
# plt.title('Test Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend(loc='upper left')
# plt.show()


# In[11]:

import gensim
model = gensim.models.Word2Vec.load("C:\\LAWHCA\\word2vec\\word2vec_wx")
print(model.most_similar(u'宾馆'))


# In[12]:

all_data=data_all
num_list=[]
for i in range(2000):
    
    num_list.append(len(all_data[i]))
print(sorted(num_list))


# In[13]:

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
        #print(i)
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


# In[14]:

print(len(res))


# In[15]:

import numpy as np
res=np.array(res)


# In[16]:

print(res.shape)


# In[17]:

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
X_train = X_train.reshape(1500, 1,MAX, 256)
X_test = X_test.reshape(500, 1,MAX, 256)


# In[5]:

print(X_train.shape)
print(y_train.shape)



# In[18]:

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
                 MAX, 256,)    # height & width
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



