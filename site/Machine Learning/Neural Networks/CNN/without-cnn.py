from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import  random
import numpy as np
import math

import multiprocessing
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

def add_layer(inputs, in_size, out_size, activation_function=None):

    Weights = tf.Variable(tf.zeros([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) )
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    elif activation_function =="Swish":
        outputs = Wx_plus_b/(1+tf.exp(-1*Wx_plus_b))
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs,Weights,biases

def nu(x_shape,y_shape):
    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, x_shape])
    ys = tf.placeholder(tf.float32, [None, y_shape])
    # add hidden layer 隱藏層
    neural_node=3000
    #l1 ,Weights1,biases1= add_layer(xs, x_shape, neural_node, activation_function=tf.nn.relu)


    # add output layer 輸出層
    prediction ,Weights2,biases2= add_layer(xs, x_shape, y_shape, activation_function=tf.nn.softmax)
    kkk=tf.log(prediction)

    # the error between prediction and real data
    """
    loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=1))
    Accuracy:91.66% 
    
    loss = tf.reduce_mean(-tf.reduce_sum(ys * prediction, reduction_indices=1))
    Accuracy :90.66
    
    loss=tf.square(ys-prediction)
    Accuracy :30%
    """
    #loss = tf.reduce_mean(-tf.reduce_sum(ys * prediction, reduction_indices=1))

    loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=1))
    #loss=tf.square(ys-prediction)


    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    # important step
    config = tf.ConfigProto(
      device_count={'GPU': 0}
    )
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)
    with tf.name_scope('Accuracy'):
      correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
      acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      tf.summary.scalar('Accuracy', acc)
    return  sess,train_step,xs,ys,prediction,acc,loss,kkk



# plot the real data



mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x_test = mnist.test.images
y_test = mnist.test.labels

# start
sess,train_step,xs,ys,prediction,acc,loss,kkk=nu(784,10)




for step in range(1000):

    # training
    new_batch_xs, new_batch_ys = mnist.train.next_batch(100)

    #print(new_batch_xs)
    sess.run(train_step, feed_dict={xs: new_batch_xs, ys: new_batch_ys})
    #if step % 50 ==0:
        #print(sess.run(loss, feed_dict = {xs: batch_xs, ys: batch_ys}))

print("Accuracy: ", sess.run(acc, feed_dict={xs: x_test, ys: y_test}))



