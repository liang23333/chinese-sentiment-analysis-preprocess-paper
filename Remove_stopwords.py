def remove_stopwords(x):
    with open("D:\\python code\\sentiment analysis\\chinese-sentiment--analysis-preprocess\\prepare data\\stopwords.txt","r") as f:
        a=f.read()
        u=x
        for i in a.split():
            u=u.replace(str(i),"")
        return u