#!/usr/bin/env python
'''
Seq2Seq implementation, practice
'''

import numpy as np
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq

class Decoder(seq2seq.Decoder):

    def __init__(self, batch_size, initial_input, initial_state):
        self._batch_size = batch_size
        self._initial_input = initial_input
        self._initial_state = initial_state

    @property
    def batch_size(self):
        return self._batch_size
    
    @property
    def output_dtype(self):
        return tf.float32
    
    @property
    def output_size(self):
        return 3
    
    @property
    def tracks_own_finished(self):
        return True

    def finalize(self, outputs, final_state, sequence_lengths):
        final_outputs = outputs
        final_state = final_state
        return final_outputs, final_state
    
    def initialize(self, name=None):
        finished = False
        initial_inputs = self._initial_input
        initial_state = self._initial_state
        return finished, initial_inputs, initial_state
    
    def step(self, time, inputs, state, name=None):
        outputs = state + inputs
        next_state = outputs
        next_inputs = outputs
        finished = time >= 3
        return outputs, next_state, next_inputs, finished

def main():

    initial_input = tf.constant([[1,2,3], [2,3,4]], dtype=tf.float32)
    initial_state = tf.constant([[1,2,3], [2,3,4]], dtype=tf.float32)
    decoder = Decoder(2, initial_input, initial_state)
    outputs, state, seq_lengths = seq2seq.dynamic_decode(decoder)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('outputs:', sess.run(outputs))
        print('state:', sess.run(state))
        print('seq_lengths:', sess.run(seq_lengths))

if __name__ == '__main__':
    main()
