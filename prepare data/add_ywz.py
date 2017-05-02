import random
with open("D:\\python code\\sentiment analysis\\chinese-sentiment--analysis-preprocess\\neg_ywz.txt",'r') as f:
    a=f.read()
    a=a.split()
    word_dict={}
    for i in range(0,len(a),2):
        word_dict[a[i]]=a[i+1]
    a=list(word_dict.keys())
    print(len(a))
    print(a)
    for i in range(0,1000):
        fname="D:\\python code\\sentiment analysis\\chinese-sentiment--analysis-preprocess\\neg\\neg."+str(i)+".txt"
        with open(fname,'a') as f1:

            t=random.randint(0,50)
            if t>=0 and t<=30 :
                f1.write(a[t])


