max_sequence_length = 7840
pos=[]
for i in range(1000):
    fname="C:\\LAWHCA\\chinese-sentiment--analysis-preprocess\\neg\\neg."+str(i)+".txt"
    #print(fname)
    with open(fname, "r",errors="ignore") as f:
        pos.append(f.read())
        #print(pos[i])

neg=[]
for i in range(1000):
    fname="C:\\LAWHCA\\chinese-sentiment--analysis-preprocess\\pos\\pos."+str(i)+".txt"
    #print(fname)
    with open(fname, "r",errors="ignore") as f:
        neg.append(f.read())
        #print(neg[i])
        

import random
data_all=pos+neg
data_dict={}
for i in range(1000):
    data_dict[data_all[i]]=1
for i in range(1000):
    data_dict[data_all[i+1000]]=0

# print(len(data_dict))
# print(data_all[0])
random.shuffle(data_all)
# print(data_all[0])
label=[]
for i in range(2000):
    label.append(data_dict[data_all[i]])

import jieba
all_data=[]
for i in range(2000):
    seg_list=jieba.cut(data_all[i])
    seg_list=" ".join(seg_list)
    all_data.append(seg_list)
# print(all_data[0])
# print(all_data[1999])


from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
tfidf = TFIDF(min_df=2, # 最小支持度为2
           max_features=None,
           strip_accents='unicode',
           analyzer='word',
           token_pattern=r'\w{1,}',
           ngram_range=(1,1),  # 二元文法模型
           use_idf=1,
           smooth_idf=1,
           sublinear_tf=1)

tfidf.fit(all_data)
all_data = tfidf.transform(all_data)
# print(type(all_data))
pos_vec=all_data[:1000]
neg_vec=all_data[1000:]
# print(all_data.shape[0])
# print(pos_vec.shape[1])
# print("\n\n\n\n\n\n")
# print(neg_vec.shape[1])
from sklearn.decomposition import PCA
z=all_data.toarray()
print(z.shape[0])
print(z.shape[1])
#pca=PCA(n_components=max_sequence_length)
#all_data=pca.fit_transform(all_data.toarray())
print(all_data.shape[0])
print(all_data.shape[1])


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
epochs = 50
batch_size = 100

rnn_size = 10
embedding_size = 1
min_word_frequency = 10
learning_rate = 0.0005
dropout_keep_prob = tf.placeholder(tf.float32)

all_data=all_data.toarray().reshape((2000,max_sequence_length,1))
label=np.array(label)



x_train, x_test = all_data[:1500], all_data[1500:]
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



epoch_seq = np.arange(1, epochs+1)
plt.plot(epoch_seq, train_loss, 'k--', label='Train Set')
plt.plot(epoch_seq, test_loss, 'r-', label='Test Set')
plt.title('Softmax Loss')
plt.xlabel('Epochs')
plt.ylabel('Softmax Loss')
plt.legend(loc='upper left')
plt.show()

# Plot accuracy over time
plt.plot(epoch_seq, train_accuracy, 'k--', label='Train Set')
plt.plot(epoch_seq, test_accuracy, 'r-', label='Test Set')
plt.title('Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()