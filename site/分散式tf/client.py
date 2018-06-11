#python

# coding=utf-8
import tensorflow as tf

# 執行目標 Session
server_target = "grpc://localhost:2222"
logs_path = './basic_tmp'

# 指定 worker task 0 使用 CPU 運算
with tf.device("/job:worker/task:1"):
    with tf.device("/cpu:0"):
        a = tf.constant([1.5, 6.0], name='a')
        b = tf.Variable([1.5, 3.2], name='b')
        c = (a * b) + (a / b)
        d = c * a
        y = tf.assign(b, d)

# 啟動 Session
with tf.Session(server_target) as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    print(sess.run(y))