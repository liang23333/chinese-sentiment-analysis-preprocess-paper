import pickle
import jieba
import string
def get_stoplist():
    with open("D:\\python code\\sentiment analysis\\stopwords.txt","r") as f:
        line=f.readlines()
        for i in range(len(line)):
            line[i]=line[i][:-1]

        return line
def process(seg_list,stop_words):
    #print(stop_words[0][1])
    ret=[]
    for i in seg_list:
        f=1
        for j in stop_words:
            if(i==j):
                f=0
                break
        if f==1:
            ret.append(i)
    return ret

def del_punctuation(seg_list):
    for i in string.punctuation+"＂（）：~~""“”，。！？、~@#￥%……&*（）1234567890 ":
        for j in range(len(seg_list)):
            if i in seg_list[j]:
                seg_list[j]=seg_list[j].replace(i,"")
    seg=[]
    for i in seg_list:
        if i != "\n" and i != '':
            seg.append(i)
    return seg
def main():
    stop_words=get_stoplist()
    f1=open('data1.pkl','wb')
    data=[]
    s1="D:\\python code\\sentiment analysis\\neg\\neg."
    for i in range(1000):
        sname=s1+str(i)+".txt"
        #print(sname)
        with open(sname,"r",encoding="gbk",errors='ignore') as f:
            s=f.read()
            seg_list=list(jieba.cut(s))
            #print(seg_list)
            seg_list=del_punctuation(seg_list)
            #print(seg_list)
            seg_list_final=process(seg_list,stop_words)
            #print(seg_list_final)
            data.append(seg_list_final)
    pickle.dump(data,f1)
    f1.close()
    data2=[]
    f2=open('data2.pkl','wb')
    s2="D:\\python code\\sentiment analysis\\pos\\pos."
    for i in range(1000):
        sname=s2+str(i)+".txt"
        #print(sname)
        with open(sname,"r",encoding="gbk",errors='ignore') as f:
            s=f.read()
            seg_list=list(jieba.cut(s))
            #print(seg_list)
            seg_list=del_punctuation(seg_list)
            #print(seg_list)
            seg_list_final=process(seg_list,stop_words)
            #print(seg_list_final)
            data2.append(seg_list_final)
    pickle.dump(data2,f2)
    f2.close()


if __name__=="__main__":
    main()
    
