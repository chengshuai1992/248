# https://blog.csdn.net/zhq0808/article/details/78482266
# 这部分是读取列表中的图片，首先是把图片resize成227*227
#
#
#
###########################################################
import read_cifar10_data
import numpy as np
import tensorflow as tf
import os
import cv2

"""
输入文件队列：这里是处理文件输入并且最后把数据整理成一个batch
tf.train.match_filenames_once试图读取多个文件 使用正则表达式获得文件名
tf.trian.string_input_producer 对文件列表创建输入队列
在这里要做的工作是生成Tfrecord文件，然后可以实现把文件组成batch
"""

config = {
    'num_shards': 10,
    'instances_per_shard': 5000,
}
Out_DIR = "../Data/cifar-10-record/"


def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def data_resize(train):
    data_image = []

    if (train):
        images, labels = read_cifar10_data.load_cifar_trian_data()
        for image in images:
            img = cv2.resize(image, (227, 227))
            data_image.append(img)
        print("train data read done!")
        return data_image, labels

    else:
        images, labels = read_cifar10_data.load_cifar_test_data()
        for image in images:
            img = cv2.resize(image, (227, 227))
            data_image.append(img)
        print("test  read done!")
        return data_image, labels


def writer_TFrecord():
    images, labels = data_resize(train=True)
    print(len(images))
    print(len(labels))

    if (len(images) == len(labels)):
        for i in range(int(config['num_shards'])):
            output_filename = "%s-%.5d-of-%.5d" % ("train", i, config['num_shards'])
            out_file = os.path.join(Out_DIR, output_filename)
            writer = tf.python_io.TFRecordWriter(out_file)
            for j in range(config['instances_per_shard']):
                image_raw = images[i * config['instances_per_shard'] + j].tobytes()
                label = labels[i * config['instances_per_shard'] + j]
                data = tf.train.Example(features=tf.train.Features(feature={
                    "image": _bytes_feature(image_raw),
                    "label": _int64_feature(label)
                }))

                writer.write(data.SerializeToString())

            writer.close()
    print("train data done!")
    test_images, test_labels = data_resize(train=False)

    if (len(test_images) == len(test_labels)):
        print(len(test_labels))
        print(len(test_images))
        output_filename = "%s-%.5d-of-%.5d" % ("test", 0, 1)
        out_file = os.path.join(Out_DIR, output_filename)
        writer = tf.python_io.TFRecordWriter(out_file)
        for j in range(len(test_images)):
            image_raw = test_images[j].tobytes()
            label = test_labels[j]
            data = tf.train.Example(features=tf.train.Features(feature={
                "image": _bytes_feature(image_raw),
                "label": _int64_feature(label)
            }))

            writer.write(data.SerializeToString())

        writer.close()
    print("test data done!")


if __name__ == '__main__':
    writer_TFrecord()