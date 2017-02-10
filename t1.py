import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1
# x_data=np.random.rand(100).astype(np.float)
# y_data=x_data*0.1+0.3
#
# Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))
# biases=tf.Variable(tf.zeros([1]))
#
# y=Weights*x_data+biases
#
# loss=tf.reduce_mean(tf.square(y-y_data))
# optimizer=tf.train.GradientDescentOptimizer(0.5)
# train=optimizer.minimize(loss)
#
# init = tf.global_variables_initializer()
#
# sess = tf.Session()
# sess.run(init)
#
# for step in range(50):
#     sess.run(train)
#     if step%10==0:
#         print(step,sess.run(Weights), sess.run(biases))


# 2
# state=tf.Variable(0,name='counter')
# one=tf.constant(1)
#
# new_value=tf.add(state,one)
# update=tf.assign(state,new_value)
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     for _ in range(3):
#         sess.run(update)
#         print(sess.run(state))

# 3
# input1=tf.placeholder(tf.float32)
# input2=tf.placeholder(tf.float32)
#
# output=tf.mul(input1,input2)
#
# with tf.Session() as sess:
#     print(sess.run(output,feed_dict={input1:[7.0],input2:[2.0]}))

# 4
def add_layer(inputs, in_size, out_size, name,activeation_function=None):
    layer_name="layer%s"%name
    with tf.name_scope(layer_name):
        with tf.name_scope("weights"):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name+'/weights',Weights)
        with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([1, out_size])+0.01)
            tf.summary.histogram(layer_name+'/biases',biases)
        with tf.name_scope("Wx_plus_b"):
            Wx_plus_biases = tf.add(tf.matmul(inputs, Weights),biases)
        if activeation_function is None:
            outputs = Wx_plus_biases
        else:
            outputs = activeation_function(Wx_plus_biases)
            tf.summary.histogram(layer_name+'/outputs',outputs)
        return outputs

#real data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# placeholder for inputs to network
with tf.name_scope("inputs"):
    xs = tf.placeholder(tf.float32, [None, 1], name="x_input")
    ys = tf.placeholder(tf.float32, [None, 1], name="y_input")

#hidden layer
l1 = add_layer(xs, 1, 10,"Fst", tf.nn.relu)
#output layer
prediction = add_layer(l1, 10, 1,"Snd", activeation_function=None)

#the error between prediciton and real data
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    tf.summary.scalar("loss",loss)

with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(0.3).minimize(loss)


sess = tf.Session()

merged= tf.summary.merge_all()
writer =tf.summary.FileWriter("logs/",sess.graph)

###matplotlib##
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.scatter(x_data, y_data)
# plt.ion()
# plt.show()

init = tf.global_variables_initializer()
sess.run(init)
#fd
for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        ####plt
        # print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        # prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        # plt.pause(0.1)
        # ax.lines.remove(lines[0])
        ##tensorboard
        result=sess.run(merged,feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result,i)
