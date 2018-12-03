#!/usr/bin/env python
'''
MNIST classification with logistic regression
'''

from dataset.mnist import get as get_mnist
import numpy as np
import tensorflow as tf
import progressbar as pbar

def mini_batch(n, *arg):
    for l in range(0, len(arg[0]), n):
        yield tuple(k[l:l+n] for k in arg)

def main():
    '''
    main function
    '''

    train_x, train_y = get_mnist(test=False)
    test_x, test_y = get_mnist(test=True)

    train_x = np.reshape(train_x, (-1, 28, 28, 1))
    test_x = np.reshape(test_x, (-1, 28, 28, 1))

    with tf.Session() as sess:

        # Build model
        input_x = tf.placeholder(tf.float32, (None, 28, 28, 1), 'input_x')
        label_y = tf.placeholder(tf.int32, (None,), 'label_y')
        is_training = tf.placeholder(tf.bool)
        enc_label_y = tf.one_hot(label_y, 10)

        h_conv0 = tf.layers.conv2d(input_x, 32, (5, 5), padding='same', activation=tf.nn.relu)
        h_pool0 = tf.layers.max_pooling2d(h_conv0, (2, 2), 2)
        h_conv1 = tf.layers.conv2d(h_pool0, 64, (5, 5), padding='same', activation=tf.nn.relu)
        h_pool1 = tf.layers.max_pooling2d(h_conv1, (2, 2), 2)
        flat = tf.reshape(h_pool1, [-1, 49*64])
        h_dense0 = tf.layers.dense(flat, 1024, tf.nn.relu)
        h_dense0_drop = tf.layers.dropout(h_dense0, 0.4, training=is_training)
        logits = tf.layers.dense(h_dense0_drop, 10)
        pred_y = tf.argmax(logits, 1)

        loss = tf.losses.sparse_softmax_cross_entropy(label_y, logits)
        minimize = tf.train.AdamOptimizer().minimize(loss)

        sess.run(tf.global_variables_initializer()) # Initialize variables

        for _ in range(1000):
            # My macbook burst!
            for batch_x, batch_y in pbar.progressbar(mini_batch(1000, train_x, train_y)):
                sess.run(minimize, {input_x: batch_x, label_y: batch_y, is_training: True})
            # Test
            pred = sess.run(pred_y, {input_x: test_x, is_training: False})
            acc = np.sum(pred == test_y) / len(test_y)
            print('acc:', acc)

if __name__ == "__main__":
    main()