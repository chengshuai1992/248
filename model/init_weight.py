import tensorflow as tf
import numpy as np
import os

def alexnet_layer():
    net_data = {}
    """
    model_weights要初始化
    distorted_image 是要作变换的数据
    input_data 227*227
    output_dim 输出维度 分类是10
    根据论文从第二个卷积开始数据分成两组，在这里使用一个GPU所以修改模型，只有一组。
    """

    ### 这里还有一个数据处理步骤

    ### Conv1
    ### Output 96, kernel 11, stride 4 out 55*55
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 96], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32),
                             trainable=True, name='biases')

        net_data['conv1'] = [kernel, biases]
        ### Pool1 out 27*27

    ### Conv2
    ### Output 256, pad 2, kernel 5 27*27
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 96, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')

        net_data['conv2'] = [kernel, biases]
    ### Pool2  out 13*13

    ### Conv3
    ### Output 384, pad 1, kernel 3 out  13*13
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 384], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')

        net_data['conv3'] = [kernel, biases]
    ### Conv4
    ### Output 384, pad 1, kernel 3, out 13*13
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 384], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')

        net_data['conv4'] = [kernel, biases]
    ### Conv5
    ### Output 256, pad 1, kernel 3  out 13*13
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')

        net_data['conv5'] = [kernel, biases]
    ### Pool5  out 6*6

    ### FC6
    ### Output 4096
    with tf.name_scope('fc6') as scope:
        fc6w = tf.Variable(tf.random_normal([9216, 4096], dtype=tf.float32, stddev=1e-2), name='weights')
        fc6b = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')

        net_data['fc6'] = [fc6w, fc6b]
        ### FC7
        ### Output 4096

    with tf.name_scope('fc7') as scope:
        fc7w = tf.Variable(tf.random_normal([4096, 4096], dtype=tf.float32, stddev=1e-2), name='weights')
        fc7b = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')

        net_data['fc7'] = [fc7w, fc7b]
        ### FC8
        ### Output output_dim
    with tf.name_scope('fc8') as scope:
        ### Differ train and val stage by 'fc8' as key

        fc8w = tf.Variable(tf.random_normal([4096, 12], dtype=tf.float32, stddev=1e-2), name='weights')
        fc8b = tf.Variable(tf.constant(0.0, shape=[12],
                                       dtype=tf.float32), trainable=True, name='biases')

    return net_data


def save_model():
    model_file = "../Data/weight/model_weights"
    model = {}
    net = alexnet_layer()
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for layer in net:
            model[layer] = sess.run(net[layer])
        print("saving model to %s" % model_file)
        np.save(model_file, np.array(model))
    sess.close()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    save_model()

