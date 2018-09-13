# https://blog.csdn.net/qq_41635631/article/details/79784391
# https://blog.csdn.net/zeuseign/article/details/72773342
import pickle
import numpy as np
import cv2

Data_DIR = "../Data/cifar-10-batches-py/"


def load_batch_cifar(cifar_file):
    cifar_file = open(cifar_file, 'rb')
    batch = pickle.load(cifar_file, encoding='latin1')
    cifar_file.close()
    image = batch['data']
    label = batch['labels']
    return image, label


def load_cifar_trian_data():
    images = []
    labels = []
    for i in range(1, 6):
        file_path = Data_DIR +"data_batch_"+str(i)

        data_image, data_label = load_batch_cifar(file_path)

        if (len(data_image) == len(data_label)):
            print("train data load")
            for i in range(len(data_image)):
                image_rgb = np.array(data_image[i]).reshape(3, 1024)
                r = np.array(image_rgb[0]).reshape(32, 32)
                g = np.array(image_rgb[1]).reshape(32, 32)
                b = np.array(image_rgb[2]).reshape(32, 32)
                image = cv2.merge([b, g, r])
                images.append(image)

                labels.append(int(data_label[i]))

    return images, labels