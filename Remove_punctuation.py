import string
def remove_punctuation(x):
    import string
    with open("D:\\python code\\sentiment analysis\\chinese-sentiment--analysis-preprocess\\prepare data\\neg_ywz.txt","r",errors='ignore') as f:
        a=f.read()
        a=a.split()
    with open("D:\\python code\\sentiment analysis\\chinese-sentiment--analysis-preprocess\\prepare data\\pos_ywz.txt","r",errors='ignore') as f:
        b=f.read()
        b=b.split()

    my_dict={}

    for i in range(0,len(a),2):
        my_dict[a[i]]=a[i+1]
    for i in range(0,len(b),2):
        my_dict[b[i]]=b[i+1]
    keys=list(my_dict.keys())
    M={}
    for i in range(len(keys)):
        for j in keys[i]:
            M[j]=1
    punctuation=list(M.keys())+list(string.punctuation)
    punctuation.remove('O')
    punctuation.remove('凸')
    punctuation.remove('3')
    punctuation.remove('ｄ')
    punctuation.remove('y')
    punctuation.remove('w')
    punctuation.remove('X')
    punctuation.remove('e')
    punctuation.remove('o')
    punctuation.remove('B')
    punctuation.remove('m')
    u=x
    for i in punctuation:
        u=u.replace(str(i),"")
    return u