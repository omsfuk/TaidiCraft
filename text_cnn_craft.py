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
      embedding_size, batch_size, num_filters, filter_sizes, embeddingW=None):
        # Placeholders for input, output and dropout

        self.questions = tf.placeholder(tf.int32, [None, question_length], name="questions")
        self.answers = tf.placeholder(tf.int32, [None, answer_length], name="answers")
        self.labels = tf.placeholder(tf.float32, [None, 1], name="labels")

        text_length = question_length + answer_length
        self.text = tf.placeholder(tf.int32, [None, text_length], name="text")
        
        # print('embeddingW',type(embeddingW))
        # Embedding layer
        with tf.name_scope("Embedding"):
            if embeddingW is not None:
                self.W = tf.cast(tf.Variable(embeddingW, name="W"), tf.float32)
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

        self.text_pool = tf.concat(text_outputs, 2)
        self.text_pool_flat = tf.reshape(self.text_pool, [-1, num_filters_total])

        with tf.name_scope("full-connected"):
            W = tf.Variable(tf.truncated_normal([num_filters_total, batch_size], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[batch_size]), name="b")
            self.text_outputs = tf.matmul(self.text_pool_flat, W) + b
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(tf.diag_part(self.text_outputs), [-1, 1]), labels=self.labels))
            """
            self.loss=tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=
                        tf.reshape(
                            tf.diag_part(
                                tf.matmul(
                                    tf.transpose(self.question_outputs), self.answer_outputs)),
                                [-1, 1]), 
                labels=self.labels))
            """
        with tf.name_scope("accuracy"):
            """
            self.a = tf.round(tf.sigmoid(tf.reshape(
                            tf.diag_part(
                                tf.matmul(
                                    tf.transpose(self.question_outputs), self.answer_outputs)),
                                [-1, 1])), name=None)
            """
            self.a = tf.round(tf.sigmoid(tf.reshape(tf.diag_part(self.text_outputs), [-1, 1])), name=None)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.a, self.labels), tf.float32))

