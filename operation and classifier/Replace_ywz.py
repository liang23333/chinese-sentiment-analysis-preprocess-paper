def replace_ywz(x):
    ywz_dict={}
    with open("D:\\python code\\sentiment analysis\\chinese-sentiment--analysis-preprocess\\prepare data\\pos_ywz.txt","r") as f:
        a=f.read()
        a=a.split()
        for i in range(0,len(a),2):
            ywz_dict[a[i]]=a[i+1]
    with open("D:\\python code\\sentiment analysis\\chinese-sentiment--analysis-preprocess\\prepare data\\neg_ywz.txt","r") as f:
        a=f.read()
        a=a.split()
        for i in range(0,len(a),2):
            ywz_dict[a[i]]=a[i+1]
    u=x
    for k,v in ywz_dict.items():
        u=u.replace(k,v)
    return u