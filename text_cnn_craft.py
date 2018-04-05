# -*- encoding: utf-8 -*-
import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, vocab_size, question_length, answer_length,
      embedding_size, classes_num, num_filters, filter_sizes, embeddingW=None):
        # Placeholders for input, output and dropout

        self.questions = tf.placeholder(tf.int32, [None, question_length], name="questions")
        self.answers = tf.placeholder(tf.int32, [None, answer_length], name="answers")
        self.labels = tf.placeholder(tf.float32, [None, classes_num], name="labels")
        
        text_length = question_length + answer_length
        self.text = tf.concat((self.questions, self.answers), axis=1)
        
        # Embedding layer
        with tf.name_scope("Embedding"):
            if embeddingW is not None:
                self.W = tf.Variable(embeddingW, name="W", trainable=True)
            else:
                self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    trainable=True, name="W")

            self.textEmbedding1 = tf.nn.embedding_lookup(self.W, self.text)
            self.textEmbedding = tf.reshape(self.textEmbedding1, [-1, text_length, embedding_size])
            self.textEmbedding_expanded = tf.expand_dims(self.textEmbedding, -1)

        # Create a convolution + maxpool layer for each filter size
        text_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("text-conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W2= tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W2")
                b2 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b2")
                conv2 = tf.nn.conv2d(
                    self.textEmbedding_expanded,
                    W2,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv2")
                # Apply nonlinearity
                h2 = tf.nn.tanh(tf.nn.bias_add(conv2, b2), name="tanh")
                # Maxpooling over the outputs
                pooled2 = tf.nn.max_pool(
                    h2,
                    ksize=[1, text_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool2")
                text_outputs.append(pooled2)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)

        self.text_pool = tf.concat(text_outputs, 3)
        self.text_pool_flat = tf.reshape(self.text_pool, [-1, num_filters_total])

        def cosine(x3, x4):
            x3_norm = tf.sqrt(tf.reduce_sum(tf.square(x3), axis=1))
            x4_norm = tf.sqrt(tf.reduce_sum(tf.square(x4), axis=1))
            #内积
            x3_x4 = tf.reduce_sum(tf.multiply(x3, x4), axis=1)
            return tf.divide(x3_x4, tf.multiply(x3_norm, x4_norm))

        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([num_filters_total, classes_num], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[classes_num]), name="b")
            self.text_outputs = tf.matmul(self.text_pool_flat, W) + b
            self.predictions = tf.argmax(self.text_outputs, 1, "predictions")
        with tf.name_scope("loss"):
            self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.text_outputs, labels=self.labels))
        with tf.name_scope("accuracy"):
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, tf.argmax(self.labels, 1)), tf.float32))

