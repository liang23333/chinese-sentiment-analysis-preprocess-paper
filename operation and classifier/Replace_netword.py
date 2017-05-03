import pickle
def replace_netword(x):
    with open("D:\\python code\\sentiment analysis\\chinese-sentiment--analysis-preprocess\\prepare data\\word_dict.pkl","rb") as f:
        word_dict=pickle.load(f)
    u=x
    for k,v in word_dict.items():
        u=u.replace(k,v)
    return u