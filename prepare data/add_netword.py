import random
netword_dict={}
with open("D:\\python code\\sentiment analysis\\chinese-sentiment--analysis-preprocess\\pos_netword.txt","r") as f:
    a=f.read()
    a=a.split()
    print(len(a))
    for i in range(0,38,2):
        netword_dict[a[i]]=a[i+1]
print(netword_dict)
a=list(netword_dict.keys())
for i in range(1000):
    s="D:\\python code\\sentiment analysis\\chinese-sentiment--analysis-preprocess\\pos\\pos."+str(i)+".txt"
    with open(s,"a") as f:
        t=random.randint(0,40)
        if t>=0 and t<19:
            f.write(a[t])

        


netword_dict={}
with open("D:\\python code\\sentiment analysis\\chinese-sentiment--analysis-preprocess\\neg_netword.txt","r") as f:
    a=f.read()
    a=a.split()
    print(len(a))
    for i in range(0,86,2):
        netword_dict[a[i]]=a[i+1]

a=list(netword_dict.keys())
for i in range(1000):
    s="D:\\python code\\sentiment analysis\\chinese-sentiment--analysis-preprocess\\neg\\neg."+str(i)+".txt"
    with open(s,"a") as f:
        t=random.randint(0,80)
        if t>=0 and t<43:
            f.write(a[t])

