#python
from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
##################################################################
                            #############
                            #前製處理函式區塊
                            #############
##################################################################
def isdigit(s):
    try:
        float(s)
        return True
    except:
        return False
class Miss_data:
    def deles(data):
        dele_table=[]
        for index,line in enumerate(data):
            for par in line:
                try:
                    float(par)
                except:
                    dele_table.append(index)
                    break
        data=np.delete(data,dele_table,axis=0)
        return data
class Normalization:
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    def simple(x):
        return (x-np.amin(x))/(np.amax(x)-np.amin(x))
    def notthing(x):
        return x

def help_toclass(in_label,class_number):
    #要做classfi 要轉換
    row,col=in_label.shape
    label=np.zeros(shape=[row,class_number])
    for ind,t in enumerate(in_label):
        temp=np.zeros(shape=[1,class_number])
        for clasfi in range(class_number):
            if clasfi==int(t[0]):
                temp[0][clasfi]=1

        label[ind]=temp
    return label

##################################################################
                            #############
                            #神神經函式區塊
                            #############
##################################################################

def add_layer(inputs, in_size, out_size, activation_function=None,norm=False):
    #神經層
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if norm:
        fc_mean,fc_var=tf.nn.moments(Wx_plus_b,axes=[0])
        scale=tf.Variable(tf.ones([out_size]))
        shift = tf.Variable(tf.zeros([out_size]))
        Wx_plus_b=(Wx_plus_b-fc_mean)/tf.sqrt(fc_var+0.001)
        Wx_plus_b=Wx_plus_b*scale+shift

    if activation_function is None:
        outputs = Wx_plus_b
    elif activation_function =="Swish":
        outputs = Wx_plus_b/(1+tf.exp(-1*Wx_plus_b))
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs,Weights,biases
def test_crossentropy(x_test,y_test):

    y_pre = sess.run(prediction, feed_dict={xs: x_test})
    loss_x_test = -tf.reduce_sum(y_test * tf.log(tf.clip_by_value(y_pre, 1e-10, 1.0)), 1, keep_dims=True)
    result = sess.run(loss_x_test, feed_dict={xs: x_test, ys: y_test})
    return result

def test_crossentropy_all(x_test,y_test):

    y_pre = sess.run(prediction, feed_dict={xs: x_test})
    loss_x_test_all = -tf.reduce_sum(y_test * tf.log(tf.clip_by_value(y_pre, 1e-10, 1.0)))
    result = sess.run(loss_x_test_all, feed_dict={xs: x_test, ys: y_test})
    return result
def nu(x_size,y_size):

    #類神經結構
    # define placeholder for inputs to network
    xs_train = tf.placeholder(tf.float32, [None, x_size])
    ys_train = tf.placeholder(tf.float32, [None, y_size])#因為我想classfi 0~3
    # add hidden layer 隱藏層
    neural_node=2048#3699bytes
    l1 ,Weights1,biases1= add_layer(xs_train, x_size, neural_node, activation_function="Swish",norm=True)

    l1_drop = tf.nn.dropout(l1, 0.26)
    l2, _, _ = add_layer(l1_drop, neural_node, neural_node, activation_function=tf.nn.relu,norm=True)
    l2_drop = tf.nn.dropout(l2, 0.26)
    # add output layer 輸出層
    prediction ,Weights2,biases2= add_layer(l2_drop, neural_node, y_size, activation_function=tf.nn.softmax,norm=True)
    # 這個在搞cross entropy 後面tf.clip_by_value就是防止讓它為零 log(0) 會引發nan問題 無法訓練
    #來源： https://stackoverflow.com/questions/33712178/tensorflow-nan-bug/33713196#33713196
    loss = -tf.reduce_sum(ys_train*tf.log(tf.clip_by_value(prediction,1e-10,1.0)))

    loss_x = -tf.reduce_sum(ys_train * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)),1,keep_dims=True)

    #loss=tf.square(ys-prediction)
    train_step = tf.train.RMSPropOptimizer(0.005).minimize(loss)



    # important step
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    return  sess,train_step,xs_train,ys_train,prediction,loss,loss_x
##################################################################
                            #############
                            #繪圖
                            #############
##################################################################

def rand_color():
    ans=u"0x"+hex(random.randint(0, 255))[2:] + hex(random.randint(0, 255))[2:] + hex(random.randint(0, 255))[2:]
    print(ans)
    return ans
##################################################################
                        #主程式區段#
##################################################################

#################
#Pre-processing
#################

#初始設定
#*****設定******
pred=13#跟此程式說要預測的col是哪個
#讀入資料
f = open('data.txt', 'r')
lines_data=f.read().split("\n")
raw_data=[]
for line in lines_data:
    li = []
    for par in line.split(","):
        li.append(par)
    raw_data.append(li)
raw_data=np.array(raw_data)


#################
# Missing data處理
###########################
print("正在做資料遺失處理...")
raw_data=Miss_data.deles(raw_data)
raw_data=raw_data.astype(float)

print("ok!")
#################

#################
# Data normalization
#################
print("正在做資料正規化...")
row,col=raw_data.shape
norm_data=np.empty(shape=[row, col])
for co in range(col):
    if co!=pred:
        r_d=Normalization.simple(raw_data[:, [co]])

    else:
        r_d=raw_data[:, [co]]
    for ro in range(row):
        norm_data[ro][co] = r_d[ro][0]
print("ok!")
#################



##################################
#Training-and-Testing-Set-Separation
##################################
#################
# Randomly split dataset
#################
#設定多少%
print("正在做資料分割train&test...")
train_pre=85##設定
test_pre=100-train_pre


test_table=[]
train_table=[]
for i in range(int(row/test_pre)):
    row,_=norm_data.shape
    test_table.append(random.randint(0,row-1))
for i in range(row):
    if i not in test_table:
        train_table.append(i)
#把原始資料分開兩群了
test=norm_data[test_table]
train=norm_data[train_table]
#再來要分開 x , y 資料
x_col=[]
for i in range(col):
    if i !=pred:
        x_col.append(i)
train_data=train[:,x_col]
train_label=train[:,[pred]]
test_data=test[:,x_col]
test_label=test[:,[pred]]
#幫忙把y從數值轉成陣列 例如 y=2,y=[0,0,1,0] or y=3,y=[0,0,0,1]
class_number=2
for index,t in enumerate(train_label):
    if t !=0:
        train_label[index]=1
for index,t in enumerate(test_label):
    if t !=0:
        test_label[index]=1
train_label=help_toclass(train_label,class_number)
test_label=help_toclass(test_label,class_number)
#print(train_label)
print("ok!")

#################
#Training-algorithm & Testing
#################
print("開始做機器學習")


# 繪圖初始化

#畫train data entropy
fig = plt.figure()
fig.canvas.set_window_title('Train data')
fig.suptitle("Train data: cross entropy & parameter")
fig.subplots_adjust(hspace=0.4, wspace=0.4)
x_label=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
y_label="cross entropy"
color_table=["#483D8B","#CD5C5C","#006400","#20B2AA","#4682B4","#800000","#0000CD","#DDA0DD","#008080","#556B2F","#DC143C","#BDB76B","#B22222"]
ax=[]
for i in range(13):
    ax.append(fig.add_subplot(4,4,i+1))
    ax[i].set_xlabel(x_label[i])
    ax[i].set_ylabel(y_label)
    ax[i].autoscale(axis='y')
train_step_image=fig.add_subplot(4,4,14)
train_step_image.set_xlabel("train step")
train_step_image.set_ylabel("cross entropy")
#畫test data entropy
fig_test_entropy = plt.figure()
fig_test_entropy.canvas.set_window_title('Test data')
fig_test_entropy.suptitle("Test data: cross entropy & parameter")
fig_test_entropy.subplots_adjust(hspace=0.4, wspace=0.4)
test_entropy=[]
for i in range(13):
    test_entropy.append(fig_test_entropy.add_subplot(4,4,i+1))
    test_entropy[i].set_xlabel(x_label[i])
    test_entropy[i].set_ylabel(y_label)
    test_entropy[i].autoscale(axis='y')
test_step_image=fig_test_entropy.add_subplot(4,4,14)
test_step_image.set_xlabel("train step")
test_step_image.set_ylabel("cross entropy")
 

fig2 = plt.figure()
fig2.canvas.set_window_title('Accuracy')
accuracy_image=fig2.add_subplot(1,1,1)
accuracy_image.set_xlabel("train step")
accuracy_image.set_ylabel("accuracy")
accuracy_image.plot(0,0,"#5F9EA0", markersize=5, marker='o', linestyle='None')
accuracy_image.plot(0,0,"#483D8B", markersize=5, marker='o', linestyle='None')
accuracy_image.legend(labels=['train data accuracy','test data accuracy'],loc='best')
accuracy_image.set_title('Train & Test Accuracy')
#lines = ax[0].plot(a, a, 'r-', lw=1)


plt.ion()
plt.pause(0.5)
plt.show()


#ax[0].lines.remove(lines[0])

def replot(ax,x_data,loss,color_table):
    line=[]
    for index,subplot in enumerate(ax):
        line.append(subplot.plot(x_data[:,index], loss, color_table[index], markersize=2,marker='o',linestyle='None'))
    return line

def deline(ax,line):
    for index,i in enumerate(line):
        try:
            ax[index].lines.remove(i[0])
        except:
            pass

def compute_accuracy(v_xs, v_ys):
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


# start
sess,train_step,xs,ys,prediction,loss,loss_x=nu(13,2)
line_train=[]
line_test=[]
for i in range(100000):
    # training
    sess.run(train_step, feed_dict={xs: train_data, ys: train_label})
    if i % 20 == 0:
        # to visualize the result and improvementd

        prediction_value = sess.run(prediction, feed_dict={xs: test_data})
        loss_x_ok=sess.run(loss_x, feed_dict={xs: train_data, ys: train_label})
        loss_all = sess.run(loss, feed_dict={xs: train_data, ys: train_label})
        #print(prediction_value)
        #畫準確度 train data
        print("train accuracy",compute_accuracy(train_data,train_label),"test accuracy",compute_accuracy(test_data,test_label))
        deline(ax,line_train)
        line_train=replot(ax,train_data,loss_x_ok,color_table)
        #畫準確度 test data
        deline(test_entropy, line_test)
        line = replot(test_entropy, test_data, test_crossentropy(test_data,test_label), color_table)



        train_step_image.plot(i, loss_all,"#5F9EA0", markersize=5, marker='o', linestyle='--')

        test_step_image.plot(i, test_crossentropy_all(test_data,test_label),"#5F9EA0", markersize=5, marker='o', linestyle='--')
        #畫train & test 準確度
        accuracy_image.plot(i, compute_accuracy(train_data,train_label),"#5F9EA0", markersize=5, marker='o', linestyle='None')
        accuracy_image.plot(i, compute_accuracy(test_data,test_label), "#483D8B", markersize=5, marker='o', linestyle='None')
        #print(loss_x)

        # plot the prediction
        #lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(0.000000001)

exit(1)