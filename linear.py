#!/usr/bin/env python
'''
Linear regression
'''
import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import progressbar as pbar
import click

@click.command()
@click.option('--epoch', default=1000, help='Training epoch')
@click.option('--func', default='sin', help='Target function')
@click.option('--activation', default='tanh', help='Activation function')
def main(epoch, func, activation):
    '''
    Main function running test
    '''

    fn_activation = {
        'sigmoid': tf.nn.sigmoid,
        'relu': tf.nn.relu,
        'tanh': tf.nn.tanh,
        'none': None,
    }[activation]

    fn_target = {
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'square': np.square,
    }[func]

    data_x = np.expand_dims(np.arange(0.0, 10.0, 0.05), -1)
    data_y = fn_target(data_x)

    with tf.Session() as sess:

        input_x = tf.placeholder(tf.float32, (None, 1), 'x')
        output_y = tf.placeholder(tf.float32, (None, 1), 'y')

        hidden_0 = tf.layers.dense(input_x, 128, fn_activation)
        hidden_1 = tf.layers.dense(hidden_0, 128, fn_activation)
        pred_y = tf.layers.dense(hidden_1, 1, fn_activation)

        loss = tf.nn.l2_loss(pred_y - output_y, 'loss')
        train = tf.train.AdamOptimizer().minimize(loss)

        sess.run(tf.global_variables_initializer())
        for _ in pbar.progressbar(range(epoch)):
            sess.run(train, {input_x: data_x, output_y: data_y})
        pred_y_val = sess.run(pred_y, {input_x: data_x})

    plt.plot(
        np.reshape(data_x, (-1,)),
        np.reshape(pred_y_val, (-1,)),
    )
    plt.plot(
        np.reshape(data_x, (-1,)),
        np.reshape(data_y, (-1,)),
    )
    plt.show()

if __name__ == '__main__':
    main()  # pylint: disable=E1120
