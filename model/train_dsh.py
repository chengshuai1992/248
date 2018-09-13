import tensorflow as tf
import read_record
import train_model
import finetune_model
import os
import numpy as np

BATCH_SIZE = 200
HASHING_BITS = 12
TRAINING_STEPS = 50000 // 200
EPOCH = 100
model_file="../Data/weight/finetune_weights"
model={}


def hash_loss(image, label, alpha, m):
    D, _ ,net_model= finetune_model.alexnet_layer(image)
    w_label = tf.matmul(label, label, False, True)

    r = tf.reshape(tf.reduce_sum(D * D, 1), [-1, 1])
    p2_distance = r - 2 * tf.matmul(D, D, False, True) + tf.transpose(r)
    temp = w_label * p2_distance + (1 - w_label) * tf.maximum(m - p2_distance, 0)

    regularizer = tf.reduce_sum(tf.abs(tf.abs(D) - 1))
    d_loss = tf.reduce_sum(temp) / (BATCH_SIZE * (BATCH_SIZE - 1)) + alpha * regularizer / BATCH_SIZE
    return d_loss,net_model


def train():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    image = tf.placeholder(tf.float32, shape=[200, 227, 227, 3], name='image')
    label = tf.placeholder(tf.float32, shape=[200, 10], name='label')

    alpha = tf.constant(0.01, dtype=tf.float32, name='tradeoff')
    m = tf.constant(HASHING_BITS * 2, dtype=tf.float32, name='bi_margin')
    global_step = tf.Variable(0, name='global_step', trainable=False)

    hloss,model_paramater = hash_loss(image, label, alpha, m)
    optimizer = tf.train.AdamOptimizer(0.001, beta1=0.5).minimize(hloss, global_step=global_step)

    with tf.Session() as sess:
        images, labels = read_record.reader_TFrecord(EPOCH)
        image_batch, label_batch = read_record.next_batch(images, labels)
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for epoch in range(EPOCH):

            for i in range(TRAINING_STEPS):
                image_b, label_b = sess.run([image_batch, label_batch])
                _, loss, step = sess.run([optimizer, hloss, global_step], feed_dict={image: image_b, label: label_b})
                if (i+1) % 50 == 0:
                    print("After %d/%d training Epoch and total %d step , current batch loss is =%.8f" % (epoch + 1, EPOCH, step, loss))
            for layer in model_paramater:
                model[layer] = sess.run(model_paramater[layer])
            print("saving model to %s" % model_file)
            np.save(model_file, np.array(model))
            print("save model successful!")

        coord.request_stop()
        coord.join(threads)
    sess.close()


if __name__ == '__main__':
    train()
