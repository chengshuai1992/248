import tensorflow as tf
import numpy as np


def alexnet_layer(image):
    ###load weight_model
    print("load Alexnet model ")
    model_weight = "../Data/weight/model_weights.npy"
    net_data = np.load(model_weight).item()
    train_paramater = {}
    ### Conv1
    ### Output 96, kernel 11, stride 4 out 55*55
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(net_data['conv1'][0], name='weights')
        biases = tf.Variable(net_data['conv1'][1], name='biases')
        #       kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 96], dtype=tf.float32, stddev=1e-1), name='weights')
        #       biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32), trainable=True, name='biases')
        conv = tf.nn.conv2d(image, kernel, [1, 4, 4, 1], padding='VALID')

        out = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(out, name=scope)
        train_paramater['conv1'] = [kernel, biases]

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
        kernel = tf.Variable(net_data['conv2'][0], name='weights')
        biases = tf.Variable(net_data['conv2'][1], name='biases')
        #       kernel = tf.Variable(tf.truncated_normal([5, 5, 96, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        #       biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        conv = tf.nn.conv2d(lrn1, kernel, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(out, name=scope)
        train_paramater['conv2'] = [kernel, biases]

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
        kernel = tf.Variable(net_data['conv3'][0], name='weights')
        biases = tf.Variable(net_data['conv3'][1], name='biases')
        #       kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 384], dtype=tf.float32, stddev=1e-1), name='weights')
        #       biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
        conv = tf.nn.conv2d(lrn2, kernel, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(out, name=scope)
        train_paramater['conv3'] = [kernel, biases]

    ### Conv4
    ### Output 384, pad 1, kernel 3, out 13*13
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(net_data['conv4'][0], name='weights')
        biases = tf.Variable(net_data['conv4'][1], name='biases')
        #       kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 384], dtype=tf.float32, stddev=1e-1), name='weights')
        #       biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(out, name=scope)
        train_paramater['conv4'] = [kernel, biases]

    ### Conv5
    ### Output 256, pad 1, kernel 3  out 13*13
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(net_data['conv5'][0], name='weights')
        biases = tf.Variable(net_data['conv5'][1], name='biases')
        #       kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        #       biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(out, name=scope)
        train_paramater['conv5'] = [kernel, biases]

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
        fc6w = tf.Variable(net_data['fc6'][0], name='weights')
        fc6b = tf.Variable(net_data['fc6'][1], name='biases')
        #       fc6w = tf.Variable(tf.random_normal([shape, 4096], dtype=tf.float32, stddev=1e-2), name='weights')
        #       fc6b = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')
        pool5_flat = tf.reshape(pool5, [-1, shape])
        fc5 = pool5_flat
        fc6l = tf.nn.bias_add(tf.matmul(pool5_flat, fc6w), fc6b)
        fc6_drop = tf.nn.dropout(tf.nn.relu(fc6l), 0.5)
        fc6 = tf.nn.relu(fc6l)
        train_paramater['fc6'] = [fc6w, fc6b]

    ### FC7
    ### Output 4096
    with tf.name_scope('fc7') as scope:
        fc7w = tf.Variable(net_data['fc7'][0], name='weights')
        fc7b = tf.Variable(net_data['fc7'][1], name='biases')
        #       fc7w = tf.Variable(tf.random_normal([4096, 4096], dtype=tf.float32, stddev=1e-2), name='weights')
        #       fc7b = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')
        fc7l = tf.nn.bias_add(tf.matmul(fc6_drop, fc7w), fc7b)
        fc7_drop = tf.nn.dropout(tf.nn.relu(fc7l), 0.5)
        fc7lo = tf.nn.bias_add(tf.matmul(fc6, fc7w), fc7b)
        fc7 = tf.nn.relu(fc7lo)
        train_paramater['fc7'] = [fc7w, fc7b]

    ### FC8
    ### Output output_dim
    with tf.name_scope('fc8') as scope:
        ### Differ train and val stage by 'fc8' as key

        fc8w = tf.Variable(tf.random_normal([4096, 12], dtype=tf.float32, stddev=1e-2), name='weights')
        fc8b = tf.Variable(tf.constant(0.0, shape=[12], dtype=tf.float32), trainable=True, name='biases')
        fc8l = tf.nn.bias_add(tf.matmul(fc7_drop, fc8w), fc8b)
        fc8 = tf.nn.bias_add(tf.matmul(fc7, fc8w), fc8b)

    return fc8l, fc8, train_paramater
