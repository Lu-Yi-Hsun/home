# View more python tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""

from __future__ import print_function
import math
import numpy as np
import random
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
PI=3.14159
def radia_transform(im,m,n):
    shape = im.shape
    new_im = np.zeros(shape)
    width = shape[0]
    height = shape[1]
    lens=len(shape)
    for i in range(0,width):
        xita = 2*PI*(i)/width
        for a in range(0,height):
            x = (int)(math.floor(a * math.cos(xita)))
            y = (int)(math.floor(a * math.sin(xita)))
            new_y = (int)(m+x)
            new_x = (int)(n+y)
            if new_x>=0 and new_x<width and new_y>=0 and new_y<height:
                if lens == 3:
                    #做有彩度rgb
                    new_im[a, i, 0] = im[new_y, new_x, 0]
                    new_im[a, i, 1] = im[new_y, new_x, 1]
                    new_im[a, i, 2] = im[new_y, new_x, 2]
                else:
                    #只有黑白
                    new_im[a, i] = im[new_y, new_x]

    return new_im

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

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])/255.   # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1] )

# print(x_image.shape)  # [n_samples, 28,28,1]

## conv1 layer ##
W_conv1 = weight_variable([5,5, 1,32]) # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32

h_pool1 = max_pool_2x2(h_conv1)                                         # output size 14x14x32

## conv2 layer ##
W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64

## fc1 layer ##
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
print(x_image)
print(h_conv1)
print(h_pool1)
# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
config = tf.ConfigProto(
    device_count={'GPU': 1}
)
sess = tf.Session(config=config)

# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)




batch_xs, batch_ys = mnist.train.next_batch(100)
control=True
#control=False
bigg=10

if control:
    batch_xs_t=[]
    for i in batch_xs:
        #print(i.shape)
        im=np.reshape(i, (28, 28))
        h = im.shape[0]
        w = im.shape[1]
        
        for j in range(bigg):
            new_im3 = radia_transform(im, (random.randint(0, 28)), (random.randint(0, 28)))
            batch_xs_t=np.append(batch_xs_t, new_im3)

    batch_xs=batch_xs_t
    print(batch_xs.shape)
    batch_xs=batch_xs.reshape(int(batch_xs.shape[0]/784),784)
    print(batch_xs.shape)
    
    for i in batch_ys:
        for j in range(bigg-1):
            batch_ys=np.append(batch_ys,i)
        #batch_ys=np.append(batch_ys,i)
        
    
    
    batch_ys=batch_ys.reshape(int(batch_ys.shape[0]/10),10)
    print(batch_ys.shape)


for i in range(50):
    #batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.1})
     
    print(compute_accuracy(
        mnist.test.images[:1000], mnist.test.labels[:1000]))
