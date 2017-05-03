import re
def remove_link(x):
    s=re.findall("([http]{4}\:\/\/[a-z_-]*(\.[a-zA-Z_-]*)*(\/([\.a-zA-Z_-]|[0-9_-])*)*\s?)",x)
    t=[]
    for i in range(len(s)):
        t.append(s[i][0])
    u=x
    for i in range(len(t)):
        u=u.replace(t[i],"")
    return u