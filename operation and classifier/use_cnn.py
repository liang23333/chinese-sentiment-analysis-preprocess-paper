import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import pickle

with open("D:\\python code\\sentiment analysis\\chinese-sentiment--analysis-preprocess\\label.pkl",'rb') as f:
    label=pickle.load(f)

print(type(label))
print(label)
with open("D:\\python code\\sentiment analysis\\chinese-sentiment--analysis-preprocess\\wordvec.pkl",'rb') as f:
    wordvec=pickle.load(f)

print(type(wordvec))
print(wordvec[0])

X_train,X_test=wordvec[:1500],wordvec[1500:]
y_train,y_test=label[:1500],wordvec[1500:]