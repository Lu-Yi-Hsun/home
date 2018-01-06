import tensorflow as tf


def add_layer(inputs, in_size, out_size, activation_function=None):
    
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    elif activation_function =="Swish":
        # by google brain :https://arxiv.org/pdf/1710.05941.pdf or https://github.com/Lu-Yi-Hsun/AI/blob/master/paper/google%20brain:Swish.pdf
        outputs = Wx_plus_b/(1+tf.exp(-1*Wx_plus_b))

    else:
        outputs = activation_function(Wx_plus_b)
    return outputs,Weights,biases
