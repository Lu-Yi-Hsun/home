import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import math
import random
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


# 讀入 MNIST
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

# 設定參數
learning_rate = 0.5
training_steps = 5000
batch_size = 10
logs_path = 'TensorBoard/'
n_features = x_train.shape[1]
n_labels = y_train.shape[1]

# 建立 Feeds
with tf.name_scope('Inputs'):
    x = tf.placeholder(tf.float32, [None, n_features], name = 'Input_Data')
with tf.name_scope('Labels'):
    y = tf.placeholder(tf.float32, [None, n_labels], name = 'Label_Data')

# 建立 Variables
with tf.name_scope('ModelParameters'):
    W = tf.Variable(tf.zeros([n_features, n_labels]), name = 'Weights')
    b = tf.Variable(tf.zeros([n_labels]), name = 'Bias')

# 開始建構深度學習模型
with tf.name_scope('Model'):
    # Softmax
    prediction = tf.nn.softmax(tf.matmul(x, W) + b)
with tf.name_scope('CrossEntropy'):
    # Cross-entropy
    loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices = 1))
    tf.summary.scalar("Loss", loss)
with tf.name_scope('GradientDescent'):
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('Accuracy', acc)

# 初始化
init = tf.global_variables_initializer()

# 開始執行運算
config = tf.ConfigProto(
      device_count={'GPU': 0}
    )
sess = tf.Session(config=config)

sess.run(init)

# 將視覺化輸出
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(logs_path, graph = tf.get_default_graph())

# 訓練
batch_xs, batch_ys = mnist.train.next_batch(100)
control=True
#control=False
bigg=2

if control:
    batch_xs_t=[]
    for i in batch_xs:
        #print(i.shape)
        im=np.reshape(i, (28, 28))
        h = im.shape[0]
        w = im.shape[1]
        
        for j in range(bigg):
            new_im3 = radia_transform(im, (w * random.uniform(0, 2)), (h * random.uniform(0, 2)))
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
     
    #print(batch_xs)
     
    #print(batch_ys)
    #print (np.append(batch_ys,batch_ys[0]).reshape(int(1010/10),10))
        #print(np.reshape(i,(1,10)))
    


 

for step in range(training_steps):
    
    #print(batch_xs.shape)
    sess.run(optimizer, feed_dict = {x: batch_xs, y: batch_ys})
    #if step % 50 == 0:
        #print(sess.run(loss, feed_dict = {x: batch_xs, y: batch_ys}))
    summary = sess.run(merged, feed_dict = {x: batch_xs, y: batch_ys})
    writer.add_summary(summary, step)

print("---")
# 準確率


sess.close()