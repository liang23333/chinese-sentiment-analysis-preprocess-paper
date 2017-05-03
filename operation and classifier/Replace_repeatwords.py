import numpy as np
def replace_repeatwords(x):
    a=np.zeros((1,len(x)))
    for i in range(1,len(x)):
        if(x[i]==x[i-1]):
            a[0,i]=a[0,i-1]+1
    u=[]
    e=0
    for i in range(len(x)):
        if(a[0,i]>=3 and (i==len(x)-1 or (a[0,i+1]<a[0,i]))):
            t=int(i-a[0,i])
            u.append(str(x[t:i+1]))
    ret=x
    for i in range(len(u)):
        ret=ret.replace(u[i],u[i][0:2])
    return ret