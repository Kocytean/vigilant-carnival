import numpy as np
import tensorflow as tf

imageHeight = 128
imageWidth = 256
kernel_width = 16
numHidden = 48

x_img = tf.placeholder(tf.float32, shape=[imageHeight,imageWidth])
y_img = tf.placeholder(tf.float32, shape=[imageHeight,imageWidth])
classDiff = tf.placeholder(tf.float32)
x = tf.expand_dims(x_img, 0)
y = tf.expand_dims(y_img, 0)
x_in = tf.expand_dims(x, 3)
y_in = tf.expand_dims(y, 3)

xConvA = tf.layers.conv2d(
	inputs= x_in,
	filters = 8,
	kernel_size = kernel_width,
	strides = 2,
	activation=tf.tanh)

yConvA = tf.layers.conv2d(
	inputs= y_in,
	filters = 8,
	kernel_size = kernel_width,
	strides = 2,
	activation=tf.tanh)

xPoolA = tf.layers.max_pooling2d(
	inputs=xConvA, 
	pool_size =2, 
	strides = 2)

yPoolA = tf.layers.max_pooling2d(
	inputs=yConvA, 
	pool_size =2, 
	strides = 2)

xConvB = tf.layers.conv2d(
	inputs= xPoolA,
	filters = 16,
	kernel_size = kernel_width,
	strides = 1,
	activation=tf.tanh)

yConvB = tf.layers.conv2d(
	inputs= yPoolA,
	filters = 16,
	kernel_size = kernel_width,
	strides = 1,
	activation=tf.tanh)

xPoolB = tf.layers.max_pooling2d(
	inputs=xConvB, 
	pool_size =2, 
	strides = 2)

yPoolB = tf.layers.max_pooling2d(
	inputs=yConvB, 
	pool_size =2, 
	strides = 2)

xFlat = tf.contrib.layers.flatten(xPoolB)
yFlat = tf.contrib.layers.flatten(yPoolB)
print(yFlat.get_shape())
differ = tf.reshape(tf.subtract(xFlat,yFlat), [11,-1])
lstmIn = tf.expand_dims(differ,0)
cell = tf.nn.rnn_cell.LSTMCell(num_units=numHidden, use_peepholes = True, state_is_tuple=True)
initter = cell.zero_state(batch_size=1, dtype= tf.float32)
val1, state = tf.nn.dynamic_rnn(cell, lstmIn, dtype=tf.float32, initial_state=initter)
val = tf.transpose(val1, [1,0,2])
last = tf.gather(val, int(val.get_shape()[0])-1)
weight =  tf.Variable(tf.truncated_normal([numHidden,1]))
bias = tf.Variable(tf.constant(0.1))
prediction = tf.tanh(tf.reduce_sum(tf.matmul(last, weight)) + bias)
cross_entropy = -(classDiff * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
optimizer = tf.train.AdamOptimizer()
killEntropy = optimizer.minimize(cross_entropy)

xer = np.random.rand(128,256)
yer = np.random.rand(128,256)
inOp = tf.initialize_all_variables()
with tf.Session() as sess:
	sess.run(inOp)
	sess.run(killEntropy, {x_img: xer, y_img: yer, classDiff: 1}) #class diff positive for fake or different sign
									#negative for verified sign
