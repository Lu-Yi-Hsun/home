# PLA(Regression)
## Type Of this Sample

| Output Space | Data Label |Protocol|Input Space|
|:---:|:---:|:---:|:---:|
|Regression|Supervised|Online|Raw|

![](/images/擷取選取區域_044.png)
該範例沒有使用Activation Function 所以只能學習線性
## Backpropagation
![](/images/擷取選取區域_043.png)
##Implement PLA(Regression)
資料=10x+5
嘗試要讓類神經藉由數據x跟y學到回歸直線
學習效率：0.0000001(讓速度慢一點好顯示)

??? note "Code"
    ```python
    import  tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt
    
    x_data=np.linspace(1,11,111).astype(np.float32)
    #x_data=np.random.rand(100).astype(np.float32)
    y_data=10*x_data+5
    
    
    Weights=tf.Variable(tf.zeros([1]))
    
    biases=tf.Variable(tf.zeros([1]))
    
    
    y=Weights*x_data+biases
    
    #loss = tf.reduce_mean(tf.square(y-y_data))#用最小平方法 可以求出回歸直線 不能亂用
    
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(y-y_data)))
    #************Optimizer*********
    #****MomentumOptimizer
    #optimizer=tf.train.MomentumOptimizer(0.1,0.2)#梯度下降法 Gradient descent
    
    
    optimizer=tf.train.GradientDescentOptimizer(0.0000001)
    
    
    train =optimizer.minimize(loss)
    
    init=tf.global_variables_initializer()
    
    sess=tf.Session()
    sess.run(init)
    
    #*********** for plt
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x_data, y_data,color="green")
    lines = ax.plot(3, 3, 'r-', lw=1)
    ax.legend(labels=['prediction line','y=10x+5'],loc='best')
    plt.ion()
    plt.show()
    plt.pause(2.5)
    for step in range(20000):
        if step % 100 == 0:
            # 只是顯示參數
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            print(step, sess.run(Weights), sess.run(biases), sess.run(loss))
            #for plt
    
            lines = ax.plot(x_data, sess.run(y), 'r-', lw=1)
            plt.pause(1)
    
    
        sess.run(train)  # 真正訓練
    
    
    
    ```

    