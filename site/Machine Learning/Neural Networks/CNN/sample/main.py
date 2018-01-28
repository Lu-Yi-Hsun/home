  
 
from __future__ import print_function
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import math
#  
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def str_to_numpy(f_strr):
    strr=f_strr.read()
    strr2=strr.split("\n")
    ans=[]
    
    for i in strr2:
        strr3=i.split("\t")
        ans_temp=[]
        for j in strr3:
            try:
                ans_temp.append(float(j))
            except:
                pass
            
            #ans_temp.append(int(j))
        if ans_temp!=[]:
            ans.append(ans_temp)
    return (np.array(ans))
    

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def Swish(input):
    outputs = input/(1+tf.exp(-1*input))
    return outputs

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])#/255.   # 28x28
ys = tf.placeholder(tf.float32, [None, 46])
LearningRate=tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])
# print(x_image.shape)  # [n_samples, 28,28,1]

## conv1 layer ##
W_conv1 = weight_variable([5,5, 1,32]) # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])

#h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # normol activation function
h_conv1 = Swish(conv2d(x_image, W_conv1) + b_conv1) ##swish

h_pool1 = max_pool_2x2(h_conv1)                                         # output size 14x14x32

## conv2 layer ##
W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])

#h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_conv2 = Swish(conv2d(h_pool1, W_conv2) + b_conv2) ##swish


h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64

## fc1 layer ##
W_fc1 = weight_variable([7*7*64, 1400])
b_fc1 = bias_variable([1400])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

#h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)#normal
h_fc1 = Swish(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)##swish

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([1400, 46])
b_fc2 = bias_variable([46])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss


#print(ss)
train_step = tf.train.AdamOptimizer(LearningRate).minimize(cross_entropy)

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)
 

 


#input train data
f_train_data=open('train_data.ml','r')
f_train_label=open('train_label.ml','r')
print("讀取Train data中...")
train_xs=str_to_numpy(f_train_data)
print("讀取Train label中...")
train_ys=str_to_numpy(f_train_label)

tem_x=[]
tem_y=[]
print(train_xs[0:2,])

tem_y.append(train_ys[0])
#input test data
print("讀取Test data中...")
f_test_data=open('test_data.ml','r')
print("讀取Test label中...")
f_test_label=open('test_label.ml','r')
test_xs=str_to_numpy(f_test_data)
test_ys=str_to_numpy(f_test_label)
 


 

def fu(x):
  #return 1 / (1 + 99999999*math.exp(-15*x))
  return math.exp(-24+13*x)

print("開始做深度學習")
batch_size=1000#硬體夠好才能調高
lr=1e-4
head=0
end=batch_size
row,col=train_xs.shape
batch_index=0
for i in range(100000):
    #batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: train_xs[head:end,], ys: train_ys[head:end,],LearningRate:lr, keep_prob: 0.5})
    #要批次處理的point
    if end==row:
        head=0
        batch_index=0
    else:
        head=end
    
    if end==row:
        end=batch_size
    elif (batch_index+2)*(batch_size)>row:
        end=row
    else:
        end=(batch_index+2)*(batch_size)
    batch_index=batch_index+1
    #lr=fu((1/compute_accuracy(test_xs,test_ys)))*1e-3
    #顯示準確度
    if i % 50 == 0:
        if compute_accuracy(test_xs,test_ys)>0.97:
         lr=1e-6
        else:
         lr=1e-3
        print(compute_accuracy(test_xs,test_ys),sess.run(cross_entropy,feed_dict={xs: train_xs[head:end,], ys: train_ys[head:end,], keep_prob: 0.5}),sess.run(LearningRate, feed_dict={xs: train_xs[head:end,], ys: train_ys[head:end,],LearningRate:lr, keep_prob: 0.5}))
    
