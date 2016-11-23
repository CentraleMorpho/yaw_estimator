from datetime import datetime
import math
import time
import numpy as np
import tensorflow.python.platform
import tensorflow as tf
import vgg_input


def conv_op(input_op, name, kw, kh, n_out, dw, dh):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w",
                                 shape=[kh, kw, n_in, n_out],
                                 dtype=tf.float32, 
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw,1), padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        activation = tf.nn.relu(z, name=scope)
        return activation

def fc_op(input_op, name, n_out, reluBool=True):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w",
                                 shape=[n_in, n_out],
                                 dtype=tf.float32, 
                                 initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
        if reluBool:
        	activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
	else:
		activation = tf.add(tf.matmul(input_op, kernel),biases)
        return activation

def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)

def loss_op(logits, labels, batch_size):
    loss = tf.nn.l2_loss(logits-labels)
    return loss



def evaluate_op(logits, labels):
    # Return probabilities of yau, pitch, roll being less than 5 degrees
    yauDeltas = tf.abs(logits[:,0]-labels[:,0])
    yauAcc = tf.less_equal(yauDeltas, 0.0277778)
    yauAcc = tf.reduce_sum(tf.cast(yauAcc, tf.float32))
    yauAcc = yauAcc/tf.to_float(labels.get_shape()[0])

    pitchDeltas = tf.abs(logits[:,1]-labels[:,1])
    pitchAcc = tf.less_equal(pitchDeltas, 0.0277778)
    pitchAcc = tf.reduce_sum(tf.cast(pitchAcc, tf.float32))
    pitchAcc = pitchAcc/tf.to_float(labels.get_shape()[0])

    rollDeltas = tf.abs(logits[:,2]-labels[:,2])
    rollAcc = tf.less_equal(rollDeltas, 0.0277778)
    rollAcc = tf.reduce_sum(tf.cast(rollAcc, tf.float32))
    rollAcc = rollAcc/tf.to_float(labels.get_shape()[0])
    return yauAcc, pitchAcc, rollAcc



def inference_cifar10_vgg(input_op, training=False):

    dropout_keep_prob = 0.5 if training else 1.0
   

    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=32, dh=1, dw=1)
    conv1_2 = conv_op(conv1_1,  name="conv1_2", kh=3, kw=3, n_out=32, dh=1, dw=1)
    pool1 = mpool_op(conv1_2,   name="pool1",   kh=2, kw=2, dw=2, dh=2)


    conv2_1 = conv_op(pool1,    name="conv2_1", kh=3, kw=3, n_out=64, dh=1, dw=1)
    conv2_2 = conv_op(conv2_1,  name="conv2_2", kh=3, kw=3, n_out=32, dh=1, dw=1)
    pool2 = mpool_op(conv2_2,   name="pool2",   kh=2, kw=2, dh=2, dw=2)

    conv3_1 = conv_op(pool2,    name="conv3_1", kh=3, kw=3, n_out=64, dh=1, dw=1)
    conv3_2 = conv_op(conv3_1,  name="conv3_2", kh=3, kw=3, n_out=64, dh=1, dw=1)
    pool3 = mpool_op(conv3_2,   name="pool3",   kh=2, kw=2, dh=2, dw=2)



    # flatten
    shp = pool3.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool3, [-1, flattened_shape], name="resh1")

    # fully connected
    fc6 = fc_op(resh1, name="fc6", n_out=64)
    fc6_drop = tf.nn.dropout(fc6, dropout_keep_prob, name="fc6_drop")


    fc8 = fc_op(fc6_drop, name="fc8", n_out=3, reluBool = False)
    return fc8

def distorted_inputs(data, batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
    labels: Labels. 1D tensor of [batch_size, 3] size.
  Raises:
    ValueError: If no data_dir
  """
  if(data=='training'):
	data_dir = "trainingImagesPaths.txt"
  elif(data=='test'):
	data_dir = "validationImagesPaths.txt"
  else:
	raise Exception('Please choose training or test as first argument')
  images, paths = vgg_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=batch_size)

  return images, paths
	
