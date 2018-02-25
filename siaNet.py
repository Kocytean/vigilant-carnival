import numpy as np
import tensorflow as tf

imageHeight = 128
imageWidth = 256
kernel_width = 16



x_in = tf.placeholder(tf.float32, [1, imageWidth, imageHeight, 1])
y_in = tf.placeholder(tf.float32, [1, imageWidth, imageHeight, 1])

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

xFlat = tf.contrib.layers.flatten()(xPoolB)
yFlat = tf.contrib.layers.flatten()(yPoolB)
cross_entropy = tf.norm(tf.subtract(xFlat,yFlat))
optimizer = tf.train.AdamOptimizer()
killEntropy = optimizer.minimize(cross_entropy)