t = 83

def func(x):
    if x <= t:
        return 0.006
    elif x > t:
        return ((0.994*(x-90))/(90-t))+1
sumi = 0
for i in range(1, 91):
    sumi+=func(i)
print(sumi/90)

