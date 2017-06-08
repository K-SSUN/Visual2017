import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import cv2 as cv


with tf.device('/cpu:0'):

    with h5py.File('resized_kalph_train.hf', 'r') as hf:
        images = np.array(hf['images'])
        labels = np.array(hf['labels'])

    num_images, rows, cols = images.shape

    images_to_dilate = images.copy()

    # translate
    translatedd_images = np.zeros((19600,28,28), dtype=np.float32)
    for i in range(19600):
        M = np.float32([[1,0,3],[0,1,random.randint(-3,3)]])
        translatedd_images[i] = cv.warpAffine(images[i], M,(cols, rows))

    images = np.concatenate((images, translatedd_images), axis=0)
    translate_i_labels = labels.copy()
    labels = np.concatenate((labels, translate_i_labels), axis=0)

    num_images, rows, cols = images.shape #39200 28 28
    #print(num_images, rows, cols)

    # magnify 1.2
    magnified_images = np.zeros((19600,28,28), dtype=np.float32)
    for i in range(19600):
        M = cv.getRotationMatrix2D((cols/2, rows/2), 0, 1.2)
        magnified_images[i] = cv.warpAffine(images[i], M,(cols, rows))

    images = np.concatenate((images, magnified_images), axis=0)
    magnify_i_labels = labels.copy()
    labels = np.concatenate((labels, magnify_i_labels), axis=0)

    num_images, rows, cols = images.shape #58800 28 28
    #print(num_images, rows, cols)

    # plus 15 dgree rotate
    rotscaled_p15_imgs = np.zeros((19600,28,28), dtype=np.float32)
    for i in range(19600):
        M = cv.getRotationMatrix2D((cols/2, rows/2), 15, 1)
        rotscaled_p15_imgs[i] = cv.warpAffine(images[i], M,(cols, rows))

    images = np.concatenate((images, rotscaled_p15_imgs), axis=0)
    rotscaled_p15_i_labels = labels.copy()
    labels = np.concatenate((labels, rotscaled_p15_i_labels), axis=0)

    num_images, rows, cols = images.shape #78400 28 28
    #print(num_images, rows, cols)

    # minus 15 dgree rotate
    rotscaled_m15_imgs = np.zeros((19600,28,28), dtype=np.float32)
    for i in range(19600):
        # M = cv.getRotationMatrix2D((cols/2, rows/2), -15, 1)
        rotscaled_m15_imgs[i] = cv.warpAffine(images[i], M,(cols, rows))

    images = np.concatenate((images, rotscaled_m15_imgs), axis=0)
    rotscaled_m15_i_labels = labels.copy()
    labels = np.concatenate((labels, rotscaled_m15_i_labels), axis=0)

    num_images, rows, cols = images.shape #98000 28 28
    #print(num_images, rows, cols)

    # dilation
    dilated_imgs = np.zeros((19600,28,28), dtype=np.float32)
    index=0
    for img in images_to_dilate:
        temp_img = img
        dilated_imgs[index] = cv.dilate(temp_img,
                                            kernel=cv.getStructuringElement(cv.MORPH_RECT, (2, 2)), iterations=1)

        index += 1

    images = np.concatenate((images, dilated_imgs), axis=0)
    dilated_i_labels = labels.copy()
    labels = np.concatenate((labels, dilated_i_labels), axis=0)

    num_images, rows, cols = images.shape #117600 28 28
    #print(num_images, rows, cols)

    # 각 픽셀값 255로 나누어 0에서 1사이의 값을 가지게 함
    images = images.astype(np.float32) / 255.0

    # One-Hot Encoding # 원핫벡터로 만드는 과정
    num_labels = labels.shape[0]
    num_classes = 14
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels.ravel()] = 1

    x_input = tf.placeholder(tf.float32, [None, 784])
    y_input = tf.placeholder(tf.float32, [None, 14])

    # 테스트 데이터 읽어오기
    with h5py.File('resized_kalph_test.hf', 'r') as hf:
       test_images = np.array(hf['images'])
       test_labels = np.array(hf['labels'])
    test_num_imgs, t_rows, t_cols = test_images.shape

    test_images = test_images.astype(np.float32) / 255.0

    # One-Hot Encoding # 원핫벡터로 만드는 과정
    t_num_labels = test_labels.shape[0]
    num_classes = 14
    t_index_offset = np.arange(t_num_labels) * num_classes
    t_labels_one_hot = np.zeros((t_num_labels, num_classes))
    t_labels_one_hot.flat[t_index_offset + test_labels.ravel()] = 1

    test_images = np.reshape(test_images, (-1, 784))  # (3920, 784)
    test_labels = np.reshape(t_labels_one_hot, (-1, 14)) # (3920, 14)

    # Weight Initialization
    def weight_variable(shape):
       initial = tf.truncated_normal(shape, stddev=0.1)
       return tf.Variable(initial)

    def bias_variable(shape):
       initial = tf.constant(0.1, shape=shape)
       return tf.Variable(initial)

    # Convolution & Pooling
    def conv2d(x, W):
       return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
       return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    # 1st Convolutional Layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x_input, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    # 1st layer를 거치면 14*14*1이 된다

    # 2 nd Convolutional Layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    # 2nd layer를 거치면 7*7*1이 된다

    # 1st Fully Connected Layer
    W_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 2 nd Fully Connected Layer
    W_fc2 = weight_variable([1024, 14])
    b_fc2 = bias_variable([14])
    #y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_input,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    batchSize = 50 # 배치 개수
    batch_xs = np.zeros(dtype = np.float32, shape=(batchSize,784))
    batch_ys = np.zeros(dtype = np.float32, shape=(batchSize,14))

    print(images.size)

    reshaped_images = np.reshape(images, (-1, 784))

    # 랜덤으로 배치 생성
    def getBatch(images, labels, size):
        for i in range(size):
            idx = random.randint(0,num_images-1)
            batch_xs[i] = reshaped_images[idx]
            batch_ys[i] = labels_one_hot[idx]

    for i in range(30000):
       getBatch(images, labels_one_hot, batchSize)
       if i%100 == 0:
          train_accuracy = accuracy.eval(feed_dict={x_input:batch_xs, y_input: batch_ys, keep_prob: 1.0})
          print('step', i, 'training accuracy', train_accuracy)
       train_step.run(feed_dict={x_input: batch_xs, y_input: batch_ys, keep_prob: 0.5})
    test_accuracy = accuracy.eval(feed_dict={x_input: test_images, y_input: test_labels, keep_prob: 1.0})

    print('test accuracy', test_accuracy)



