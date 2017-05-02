import pickle
with open("D:\\python code\\sentiment analysis\\chinese-sentiment--analysis-preprocess\\fuhaobiaoqing.txt","r") as f:
    a=f.read()
    a=a.split()
    print(len(a))
    ywz_dict={}
    for i in range(0,len(a),2):
        ywz_dict[a[i]]=a[i+1]
    for k,v in ywz_dict.items():
        print(k,v)
    with open("D:\\python code\\sentiment analysis\\chinese-sentiment--analysis-preprocess\\ywz_dict.pkl","wb") as f1:
        
        pickle.dump(ywz_dict,f1)
    with open("D:\\python code\\sentiment analysis\\chinese-sentiment--analysis-preprocess\\ywz_dict.pkl","rb") as f1:
        a=pickle.load(f1)
        print(len(a))
        print(len(ywz_dict))