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

    train_x = np.reshape(train_x, (len(train_x), -1))
    test_x = np.reshape(test_x, (len(test_x), -1))

    with tf.Session() as sess:

        # Build model
        input_x = tf.placeholder(tf.float32, (None, 784), 'input_x')
        label_y = tf.placeholder(tf.int32, (None,), 'label_y')
        enc_label_y = tf.one_hot(label_y, 10)
        h0 = tf.layers.dense(input_x, 128, tf.nn.sigmoid)
        h1 = tf.layers.dense(h0, 128, tf.nn.sigmoid)
        out_y = tf.layers.dense(h1, 10, tf.nn.sigmoid)
        pred_y = tf.argmax(out_y, 1)

        loss = tf.nn.l2_loss(out_y - enc_label_y, 'loss')
        minimize = tf.train.AdamOptimizer().minimize(loss)

        sess.run(tf.global_variables_initializer()) # Initialize variables

        for _ in range(1000):
            for batch_x, batch_y in mini_batch(1000, train_x, train_y):
                sess.run(minimize, {input_x: batch_x, label_y: batch_y})
            # Test
            pred = sess.run(pred_y, {input_x: test_x})
            acc = np.sum(pred == test_y) / len(test_y)
            print('acc:', acc)

if __name__ == "__main__":
    main()