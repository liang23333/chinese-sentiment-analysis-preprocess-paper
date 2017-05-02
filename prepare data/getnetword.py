# -*- coding: UTF-8 -*-  
import sys   
import requests
import re
import pickle
from bs4 import BeautifulSoup
import urllib
import time
word_dict={}
f=open("D:\\python code\\sentiment analysis\\word_dict.txt",'a')
for i in range(5,11):
    if i==1:
        url='http://wangci.net/word.html'
    else:
        url='http://wangci.net/word_'+str(i)+'.html'
    print(url)
    req=urllib.request.urlopen(url)
    response=req.read().decode('gbk',errors='ignore')
    #response="".join(response.split())    
    print("\n\n\n\n")
    #print(response)
    p=re.compile('<td width=\"10%\">&nbsp;<a href=\"(.*?)\" target=\"_blank\">(.*?)<\/a><\/td>')
    res=p.findall(response)
    print(res)
    sum=0
    for a,b in res:
        a='http://wangci.net/'+a

        req=urllib.request.urlopen(a)
        response=req.read().decode('gbk',errors='ignore')
        #print(response)
        response="".join(response.split())
        p=re.compile('<strong>1.(.*?)<br>')
        res1=p.findall(response)
        print(b,res1)
        time.sleep(2)
        word_dict[b]=res1
        f.write(b)
        f.write('\t')
        for i in res1:

            f.write(i)
        f.write('\n')

    print(i)
    print(word_dict)
f.close()

    


