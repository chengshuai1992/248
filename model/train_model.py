###这里是Alexnet网络结构
###http://www.360doc.com/content/18/0429/06/36490684_749589931.shtml
###https://www.jianshu.com/p/916262f672e7 关于DSH的中文介绍

"""
model_weights要初始化
distorted_image 是要作变换的数据
input_data 227*227
output_dim 输出维度 分类是10
根据论文从第二个卷积开始数据分成两组，在这里使用一个GPU所以修改模型，只有一组。
"""

### 这里还有一个数据处理步骤

import tensorflow as tf
import numpy as np


def alexnet_layer(image):

    ### Conv1
    ### Output 96, kernel 11, stride 4 out 55*55
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 96], dtype=tf.float32,stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32),trainable=True, name='biases')
        conv = tf.nn.conv2d(image, kernel, [1, 4, 4, 1], padding='VALID')

        out = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(out, name=scope)


    ### Pool1 out 27*27
    pool1 = tf.nn.max_pool(conv1,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool1')

    ### LRN1
    radius = 2;
    alpha = 2e-05;
    beta = 0.75;
    bias = 1.0
    lrn1 = tf.nn.local_response_normalization(pool1,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    ### Conv2
    ### Output 256, pad 2, kernel 5 27*27
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 96, 256], dtype=tf.float32,stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),trainable=True, name='biases')
        conv = tf.nn.conv2d(lrn1, kernel, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(out, name=scope)


    ### Pool2  out 13*13
    pool2 = tf.nn.max_pool(conv2,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool2')

    ### LRN2
    radius = 2;
    alpha = 2e-05;
    beta = 0.75;
    bias = 1.0
    lrn2 = tf.nn.local_response_normalization(pool2,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    ### Conv3
    ### Output 384, pad 1, kernel 3 out  13*13
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 384], dtype=tf.float32,stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),trainable=True, name='biases')
        conv = tf.nn.conv2d(lrn2, kernel, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(out, name=scope)


    ### Conv4
    ### Output 384, pad 1, kernel 3, out 13*13
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 384], dtype=tf.float32,stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),trainable=True, name='biases')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(out, name=scope)


    ### Conv5
    ### Output 256, pad 1, kernel 3  out 13*13
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32,stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),trainable=True, name='biases')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(out, name=scope)


    ### Pool5  out 6*6*256
    pool5 = tf.nn.max_pool(conv5,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool5')

    ### FC6  6*6*256
    ### Output 4096
    with tf.name_scope('fc6') as scope:
        shape = int(np.prod(pool5.get_shape()[1:]))

        fc6w = tf.Variable(tf.random_normal([shape, 4096], dtype=tf.float32, stddev=1e-2), name='weights')
        fc6b = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')
        pool5_flat = tf.reshape(pool5, [-1, shape])
        fc5 = pool5_flat
        fc6l = tf.nn.bias_add(tf.matmul(pool5_flat, fc6w), fc6b)
        fc6_drop = tf.nn.dropout(tf.nn.relu(fc6l), 0.5)
        fc6 = tf.nn.relu(fc6l)


    ### FC7
    ### Output 4096
    with tf.name_scope('fc7') as scope:
        fc7w = tf.Variable(tf.random_normal([4096, 4096], dtype=tf.float32, stddev=1e-2), name='weights')
        fc7b = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')
        fc7l = tf.nn.bias_add(tf.matmul(fc6_drop, fc7w), fc7b)
        fc7_drop = tf.nn.dropout(tf.nn.relu(fc7l), 0.5)
        fc7lo = tf.nn.bias_add(tf.matmul(fc6, fc7w), fc7b)
        fc7 = tf.nn.relu(fc7lo)


    ### FC8
    ### Output output_dim
    with tf.name_scope('fc8') as scope:
        ### Differ train and val stage by 'fc8' as key

        fc8w = tf.Variable(tf.random_normal([4096, 12], dtype=tf.float32, stddev=1e-2), name='weights')
        fc8b = tf.Variable(tf.constant(0.0, shape=[12],dtype=tf.float32), trainable=True, name='biases')
        fc8l = tf.nn.bias_add(tf.matmul(fc7_drop, fc8w), fc8b)
        fc8 = tf.nn.bias_add(tf.matmul(fc7, fc8w), fc8b)

    return fc8l, fc8
