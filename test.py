import tensorflow as tf
from tensorflow import keras


def load_dataset():
    # Step0 准备数据集, 可以是自己动手丰衣足食, 也可以从 tf.keras.datasets 加载需要的数据集(获取到的是numpy数据)
    # 这里以 mnist 为例
    (x, y), (x_test, y_test) = keras.datasets.mnist.load_data()

    db_train = tf.data.Dataset.from_tensor_slices((x, y))
    db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # Step2 打乱数据
    db_train.shuffle(1000)
    db_test.shuffle(1000)

    # Step3 预处理 (预处理函数在下面)
    db_train.map(preprocess)
    db_test.map(preprocess)

    # Step4 设置 batch size 一次喂入64个数据
    db_train.batch(64)
    db_test.batch(64)

    # Step5 设置迭代次数(迭代2次) test数据集不需要emmm
    db_train.repeat(2)

    return db_train, db_test


def preprocess(labels, images):
    '''
    最简单的预处理函数:
        转numpy为Tensor、分类问题需要处理label为one_hot编码、处理训练数据
    '''
    # 把numpy数据转为Tensor
    labels = tf.cast(labels, dtype=tf.int32)
    # labels 转为one_hot编码
    labels = tf.one_hot(labels, depth=10)
    # 顺手归一化
    images = tf.cast(images, dtype=tf.float32) / 255
    return labels, images

print(load_dataset())
