# coding=utf8
"""
author=Aaron
python=3.5
keras=2.0.6
tensorflow=1.2.1
"""
from keras import Input
import numpy as np
from keras.layers import MaxPooling2D, UpSampling2D, Conv2D
from keras.models import Model
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# 输入数据，包括数据集的下载、读取、切分、归一化处理等。
mnist = input_data.read_data_sets('D:\data\minist\\', one_hot=True)
x_train, x_test = mnist.train.images, mnist.test.images

# 将784维转换为28*28矩阵。
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# 加入随机白噪。
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# 区间剪切，超过区间会被转成区间极值
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# 在原图和加噪声图中各选取十张绘图显示比对。
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display noise images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# 定义encoder
input_img = Input(shape=(28, 28, 1))  # (?, 28, 28, 1)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)  # (?, 28, 28, 32)
x = MaxPooling2D((2, 2), padding='same')(x)  # (?, 14, 14, 32)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)  # (?, 14, 14, 32)
encoded = MaxPooling2D((2, 2), padding='same')(x)  # (?, 7, 7, 32)

# 定义decoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)  # (?, 7, 7, 32)
x = UpSampling2D((2, 2))(x)  # (?, 14, 14, 32)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)  # (?, 14, 14, 32)
x = UpSampling2D((2, 2))(x)  # (?, 28, 28, 32)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)  # (?, 28, 28, 1)

# 选定模型的输入，decoded（即输出）的格式
auto_encoder = Model(input_img, decoded)
# 定义优化目标和损失函数
auto_encoder.compile(optimizer='sgd', loss='mean_squared_error')

# 训练
auto_encoder.fit(x_train_noisy, x_train,  # 输入输出
                 epochs=100,  # 迭代次数
                 batch_size=128,
                 shuffle=True,
                 validation_data=(x_test_noisy, x_test))  # 验证集

decoded_imgs = auto_encoder.predict(x_test_noisy)  # 测试集合输入查看器去噪之后输出。

# 在测试集合中选加噪声图和去噪图中各选取十张绘图显示比对。
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
