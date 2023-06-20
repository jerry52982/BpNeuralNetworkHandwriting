import os
import struct
import numpy as np
import shutil

#将训练的mnist训练数据板块转成csv形式
def load_mnist1(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

#将测试的mnist的测试数据板块转成csv形式
def load_mnist2(path, kind='t10k'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

X_train,y_train = load_mnist1('C:\\Users\\韦祖垚\\Desktop\\模式识别大作业韦祖垚\\data')
np.savetxt('data\\train_img.csv', X_train,
           fmt='%i', delimiter=',')        #train_img是训练的数据的图像的矩阵形式，一共784列
np.savetxt('data\\train_labels.csv', y_train,
           fmt='%i', delimiter=',')        #train_labels是训练的数据一般表示方法(0-9)，一列

X_test,y_test=load_mnist2('C:\\Users\\韦祖垚\\Desktop\\模式识别大作业韦祖垚\\data')
np.savetxt('data\\test_img.csv', X_test,
           fmt='%i', delimiter=',')        #测试集的矩阵形式
np.savetxt('data\\test_labels.csv', y_test,
           fmt='%i', delimiter=',')        #测试集的数据

