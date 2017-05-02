def remove_number(x):
    u=x
    for i in range(10):
        u=u.replace(str(i),"")
    return u
