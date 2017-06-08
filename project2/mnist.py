from tensorflow.examples.tutorials.mnist import input_data
import gzip
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

data_dir = './MNIST_data/'
mnist = input_data.read_data_sets(data_dir, one_hot=True, validation_size=5000)

def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

with open('./MNIST_data/train-images-idx3-ubyte.gz', 'rb') as f:
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)

        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)

        buf = bytestream.read(rows*cols*num_images)
        data = np.frombuffer(buf, dtype=np.uint8)

        data = data.reshape(num_images, rows, cols)
rows = 10
cols = 10

with open('./MNIST_data/train-labels-idx1-ubyte.gz', 'rb') as f:
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        num_items = _read32(bytestream)

        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)


x_input = tf.placeholder(tf.float32, [None, 784])
y_input = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x_input, W) + b

y = tf.nn.softmax(y)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x_input: batch_xs, y_input: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_input, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy,
               feed_dict={x_input: mnist.test.images, y_input: mnist.test.labels}))
