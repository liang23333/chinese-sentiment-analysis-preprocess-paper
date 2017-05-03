
# coding: utf-8

# In[1]:

pos=[]
for i in range(1000):
    fname="D:\\python code\\sentiment analysis\\chinese-sentiment--analysis-preprocess\\neg\\neg."+str(i)+".txt"
    print(fname)
    with open(fname, "r",errors="ignore") as f:
        pos.append(f.read())
        print(pos[i])


# In[2]:

neg=[]
for i in range(1000):
    fname="D:\\python code\\sentiment analysis\\chinese-sentiment--analysis-preprocess\\pos\\pos."+str(i)+".txt"
    print(fname)
    with open(fname, "r",errors="ignore") as f:
        neg.append(f.read())
        print(neg[i])


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

import jieba
all_data=[]
for i in range(2000):
    seg_list=jieba.cut(data_all[i])
    seg_list=" ".join(seg_list)
    all_data.append(seg_list)
print(all_data[0])
print(all_data[1999])


# In[15]:




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
print(type(all_data))
pos_vec=all_data[:1000]
neg_vec=all_data[1000:]
print(all_data.shape[0])
print(pos_vec.shape[1])
print("\n\n\n\n\n\n")
print(neg_vec.shape[1])


# In[16]:

print(tfidf.vocabulary_)


# In[18]:

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


# In[ ]:



