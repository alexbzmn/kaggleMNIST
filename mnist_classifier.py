from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import import_data as imp
import numpy as np
import matplotlib.pyplot as plt

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

# First Convolutional Layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second Convolutional Layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely Connected Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

sess = tf.InteractiveSession()

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

readout_func = tf.nn.softmax(y_conv + b_fc2)
sess.run(tf.initialize_all_variables())

# read features & labels
######################################
features = imp.read_features_from_csv('data/train.csv')
labels = imp.read_labels_from_csv('data/train.csv')

TRAIN_SPLIT = 0.85  # training/validation split
BATCH_SIZE = 50
TRAINING_STEPS = int(len(features) * TRAIN_SPLIT / BATCH_SIZE * 33)

# split data into training and validation sets
train_samples = int(len(features) / (1 / TRAIN_SPLIT))
train_features = features[:train_samples]
train_labels = labels[:train_samples]
validation_features = features[train_samples:]
validation_labels = labels[train_samples:]

for i in range(TRAINING_STEPS):
    batch_features, batch_labels = imp.generate_batch(train_features, train_labels, BATCH_SIZE)
    if i == 0 or (i + 1) % (TRAINING_STEPS // 33) == 0 or (i + 1) == TRAINING_STEPS:
        val_features, val_labels = imp.generate_batch(validation_features, validation_labels, BATCH_SIZE)
        train_accuracy = accuracy.eval(feed_dict={
            x: val_features, y_: val_labels, keep_prob: 1.0})
        print('step', i + 1, 'of', TRAINING_STEPS, '/ validation accuracy:', train_accuracy)
    train_step.run(feed_dict={x: batch_features, y_: batch_labels, keep_prob: 0.5})

test_features = imp.read_features_from_csv('data/test.csv', usecols=None)
readout = sess.run(y, feed_dict={x: test_features, keep_prob: 1.0})
readout = np.argmax(readout, axis=1)

readout = [np.arange(1, 1 + len(readout)), readout]
readout = np.transpose(readout)

# write to csv file
np.savetxt('submission-mnist.csv', readout, fmt='%i,%i', header='ImageId,Label', comments='')
######################################


# for i in range(20000):
#     batch = mnist.train.next_batch(50)
#     if i % 100 == 0:
#         train_accuracy = accuracy.eval(feed_dict={
#             x: batch[0], y_: batch[1], keep_prob: 1.0})
#         print("step %d, training accuracy %g" % (i, train_accuracy))
#     train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
#
# print("test accuracy %g" % accuracy.eval(feed_dict={
#     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
