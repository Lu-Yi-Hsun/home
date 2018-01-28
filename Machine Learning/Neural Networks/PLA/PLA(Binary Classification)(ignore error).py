import matplotlib.pyplot as plt
import numpy as np
import random
#網路上找的dataset 可以線性分割

dataset = np.array([
((1, 1, 5), -1),
((1, 2, 4), -1),
((1, 3, 3), -1),
((1, 4, 2), -1),
((1, 1, 6), 1),
((1, 2, 5), 1),
((1, 3, 4), 1),
((1, 4, 3), 1)])

#判斷有沒有分類錯誤，並列印錯誤率

def check_error(w, dataset,error_persent):


    result = None
    error = 0
    for x, s in dataset:
        x = np.array(x)
        print(w.T.dot(x))
        if int(np.sign(w.T.dot(x))) != s:
            #帶入ax+by+c=0 如果符號不相等代表有錯誤

            # T transpose
            result =  x, s
            error += 1
    print  ("error=%s/%s" % (error, len(dataset)))
    if (error*100/len(dataset))<error_persent:
        return None
    else:
        return result

#PLA演算法實作
#Cyclic Perceptron Learning Algorithm
def pla(dataset,error_persent):

    #ax+by+c=0 線性方程式
    w = np.zeros(3)
    index=0
    while check_error(w, dataset,error_persent) is not None:


        x, s = check_error(w, dataset,error_persent)
        w += (s) * x
        #fig by algorithm(1).md
        index=index+1

    return w


def print_image(w):

    #畫圖
    ps = [v[0] for v in dataset]
    value = [v[1] for v in dataset]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    #111 is control code 1
    #These are subplot grid parameters encoded as a single integer. For example, "111" means "1x1 grid, first subplot" and "234" means "2x3 grid, 4th subplot".
    #dataset前半後半已經分割好 直接畫就是
    index=0

    max_x=ps[0][1]
    min_x=ps[0][1]
    for v in value:
        #print(index)
        if v>0:
            ax1.scatter(ps[index][1],ps[index][2], c='b', marker="o")
        elif v<0:
            ax1.scatter(ps[index][1],ps[index][2] , c='r', marker="x")
        else:
            pass
        if max_x<ps[index][1]:
            max_x=ps[index][1]
        if min_x>ps[index][1]:
            min_x=ps[index][1]
        index=index+1

    l = np.linspace(min_x-1,max_x+1)
    #define the line x-axis size
    a,b = -w[1]/w[2], -w[0]/w[2]
    #a=斜率 b常數
    ax1.plot(l, a*l + b, 'b-')

    plt.show()
    plt.close()

a = np.array([[(1, 1, 5) ,-1]])


dataset=np.array([[(1, 1, 5) ,-1]])
k=30
arr=[]
for i in range(k*3):

    if (i+1)%3==0:
        dataset=np.concatenate((dataset, np.array([[(1, arr[0], arr[1]) ,np.sign(random.uniform(-100, 100))]])))
        arr=[]
    else:
        arr.append(random.uniform(-100, 100))

print (dataset)
#print(dataset)
#exit(1)
w = pla(dataset,88)

print_image(w)