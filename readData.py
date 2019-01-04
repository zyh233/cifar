# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 17:21:36 2019

@author: zhangyonghong
"""
import numpy as np
def unpickle(file):
    '''
    :param file: 输入文件是cifar-10的python版本文件，一共五个batch，每个batch10000个32×32大小的彩色图像，一次输入一个
    :return: 返回的是一个字典，key包括b'filename',b'labels',b'data'
    '''
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def onehot(labels):
    '''
    :param labels: 输入标签b'labels'，大小为10000的列表，每一个元素直接是对应图像的类别，需将其转换成onehot格式
    :return: onehot格式的标签，10000行，10列，每行对应一个图像，列表示10个分类，该图像的类所在列为1，其余列为0
    '''
    #  先建立一个全0的，10000行，10列的数组
    onehot_labels = np.zeros([len(labels), 10])
    # 这里的索引方式比较特殊，第一个索引为[0, 1, 2, ..., len(labels)]，
    # 第二个索引为[第1个的图像的类, 第2个的图像的类, ..., 最后一个的图像的类]
    # 即将所有图像的类别所对应的位置改变为1
    onehot_labels[np.arange(len(labels)), labels] = 1
    return onehot_labels
def readData():
# 反序列化cifar-10训练数据和测试数据
    data1 = unpickle(r'.\cifar-10-batches-py\data_batch_1')
    data2 = unpickle(r'.\cifar-10-batches-py\data_batch_2')
    data3 = unpickle(r'.\cifar-10-batches-py\data_batch_3')
    data4 = unpickle(r'.\cifar-10-batches-py\data_batch_4')
    data5 = unpickle(r'.\cifar-10-batches-py\data_batch_5')
    test_data = unpickle(r'.\cifar-10-batches-py\test_batch')
    
    #data1[b'data'] = data1[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).reshape(10000, -1)
    #data2[b'data'] = data2[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).reshape(10000, -1)
    #data3[b'data'] = data3[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).reshape(10000, -1)
    #data4[b'data'] = data4[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).reshape(10000, -1)
    #data5[b'data'] = data5[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).reshape(10000, -1)
    #test_data[b'data'] = test_data[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).reshape(10000, -1)
    
    # 合并5个batch得到相应的训练数据和标签
    x_train = np.concatenate((data1[b'data'], data2[b'data'], data3[b'data'], data4[b'data'], data5[b'data']), axis=0)
    y_train = np.concatenate((data1[b'labels'], data2[b'labels'], data3[b'labels'], data4[b'labels'], data5[b'labels']), axis=0)
    y_train = onehot(y_train)
     
    # 得到测试数据和测试标签
    x_test = test_data[b'data']
    y_test = onehot(test_data[b'labels'])
    return x_train, y_train, x_test, y_test

def main():
    [x, y, x_t, y_t] = readData()
    print(x.shape)
    print(y[1:100])

if __name__ == '__main__':
    main()
