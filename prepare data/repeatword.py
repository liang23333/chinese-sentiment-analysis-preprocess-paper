import random
for i in range(0,1000):
    fname="D:\\python code\\sentiment analysis\\chinese-sentiment--analysis-preprocess\\pos\\pos."+str(i)+".txt"
    with open(fname,"r",errors="ignore") as f:
        a=f.read()
        n=random.randint(0,3)
        for j in range(n):
            t=random.randint(0,len(a)-1)
            s=a[:t]
            for num in range(0,10):
                s=s+a[t]
            s=s+a[t:]
            a=s
        print(a)

    with open(fname,'w') as f:
        f.write(a)

        

