import pickle
with open("D:\\python code\\sentiment analysis\\word_dict.txt","r") as f:
    content=f.read()
    word_dict={}
    content=content.split()
    print(len(content))
    for i in range(0,490,2):
        word_dict[content[i]]=content[i+1]
    print(len(word_dict))

    print("\n\n\n\n\n\n\n\n")
    with open("D:\\python code\\sentiment analysis\\word_dict.pkl",'wb') as f1:
        pickle.dump(word_dict,f1)
    with open("D:\\python code\\sentiment analysis\\word_dict.pkl",'rb') as f1:
        a=pickle.load(f1)
    print(len(a))
