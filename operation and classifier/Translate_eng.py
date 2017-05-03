import pickle
a="123"
english_fname=[]
name="D:\\python code\\sentiment analysis\\chinese-sentiment-analysis-preprocess\\chinese-sentiment-analysis-preprocess-paper\\"
sum=0

english=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']


def Judge(fname,z):
    for i in range(52):
        if english[i] in z:
            return 1
    return 0


print(len(english))
for i in range(1000):
    fname=name+"pos\\pos."+str(i)+".txt"
    with open(fname,"r",errors='ignore') as f:
        z=f.read()
        if Judge(fname,z):
            english_fname.append(fname)
            sum=sum+1
        

for i in range(1000):
    fname=name+"neg\\neg."+str(i)+".txt"
    with open(fname,"r",errors='ignore') as f:
        z=f.read()
        z=f.read()
        if Judge(fname,z):
            english_fname.append(fname)
            sum=sum+1

print(sum)

with open("D:\\python code\\sentiment analysis\\chinese-sentiment-analysis-preprocess\\chinese-sentiment-analysis-preprocess-paper\\english_fname.pkl","wb") as f:
    pickle.dump(english_fname,f)
english2chinese_dict={}
import requests
import random
import time
for i in range(sum):
    fname=english_fname[i]
    with open(fname,"r",errors='ignore') as f:
        z=f.read()
        url1='http://139.199.209.106/trans/tencent.action?from=en&to=zh&query='+z
        url2='http://139.199.209.106/trans/youdao.action?from=en&to=zh&query='+z
        url3='http://139.199.209.106/trans/baidu.action?from=en&to=zh&query='+z
        t = random.randint(1,3)
        if t == 1:
            english2chinese_dict[fname]=requests.get(url=url1).text
        elif t == 2:
            english2chinese_dict[fname]=requests.get(url=url2).text
        else:
            english2chinese_dict[fname]=requests.get(url=url3).text
        time.sleep(5)
        print(fname)

with open("D:\\python code\\sentiment analysis\\chinese-sentiment-analysis-preprocess\\chinese-sentiment-analysis-preprocess-paper\\english2chinese.pkl","wb") as f:
    pickle.dump(english2chinese_dict,f)
        
