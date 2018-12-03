#!/usr/bin/env python
'''
Iris classification with logistic regression
'''

from dataset.iris import get as get_iris
import numpy as np
import tensorflow as tf
import progressbar as pbar

def main():
    '''
    main function
    '''

    train_x, train_y = get_iris(test=False)
    test_x, test_y = get_iris(test=True)

    with tf.Session() as sess:

        # Build model
        input_x = tf.placeholder(tf.float32, (None, 4), 'input_x')
        label_y = tf.placeholder(tf.int32, (None,), 'label_y')
        enc_label_y = tf.one_hot(label_y, 3)
        h0 = tf.layers.dense(input_x, 128, tf.nn.sigmoid)
        h1 = tf.layers.dense(h0, 128, tf.nn.sigmoid)
        out_y = tf.layers.dense(h1, 3, tf.nn.sigmoid)
        pred_y = tf.argmax(out_y, 1)

        loss = tf.nn.l2_loss(out_y - enc_label_y, 'loss')
        minimize = tf.train.AdamOptimizer().minimize(loss)

        sess.run(tf.global_variables_initializer()) # Initialize variables

        # Train variables
        for _ in pbar.progressbar(range(10000)):
            sess.run(minimize, {input_x: train_x, label_y: train_y})

        # Test
        pred = sess.run(pred_y, {input_x: test_x})
        acc = np.sum(pred == test_y) / len(test_y)
        print('acc:', acc)

if __name__ == "__main__":
    main()