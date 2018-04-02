# -*- encoding: utf-8 -*-
import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, vocab_size, question_num, answer_num, question_length, answer_length,
      embedding_size, batch_size, num_filters, filter_sizes, embeddingW=None):
        # Placeholders for input, output and dropout

        self.questions = tf.placeholder(tf.int32, [None, question_length], name="questions")
        self.answers = tf.placeholder(tf.int32, [None, answer_length], name="answers")
        self.labels = tf.placeholder(tf.float32, [None, 1], name="labels")
        
        # print('embeddingW',type(embeddingW))
        # Embedding layer
        with tf.name_scope("Embedding"):
            if embeddingW is not None:
                self.W = tf.Variable(embeddingW, name="W", trainable=True)
            else:
                self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    trainable=True, name="W")


            self.questionEmbedding1 = tf.nn.embedding_lookup(self.W, self.questions)
            self.questionEmbedding = tf.reshape(self.questionEmbedding1, [-1, question_length, embedding_size])
            self.questionEmbedding_expanded = tf.expand_dims(self.questionEmbedding, -1)

            self.answerEmbedding1 = tf.nn.embedding_lookup(self.W, self.answers)
            self.answerEmbedding = tf.reshape(self.answerEmbedding1, [-1, answer_length, embedding_size])
            self.answerEmbedding_expanded = tf.expand_dims(self.answerEmbedding, -1)

        # Create a convolution + maxpool layer for each filter size
        question_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("question-conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W2= tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W2")
                b2 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b2")
                conv2 = tf.nn.conv2d(
                    self.questionEmbedding_expanded,
                    W2,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv2")
                # Apply nonlinearity
                h2 = tf.nn.tanh(tf.nn.bias_add(conv2, b2), name="tanh")
                # Maxpooling over the outputs
                pooled2 = tf.nn.max_pool(
                    h2,
                    ksize=[1, question_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool2")
                print(tf.shape(pooled2))
                question_outputs.append(pooled2)

        answer_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("answer-conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W2= tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W2")
                b2 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b2")
                conv2 = tf.nn.conv2d(
                    self.answerEmbedding_expanded,
                    W2,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv2")
                # Apply nonlinearity
                h2 = tf.nn.tanh(tf.nn.bias_add(conv2, b2), name="tanh")
                # Maxpooling over the outputs
                pooled2 = tf.nn.max_pool(
                    h2,
                    ksize=[1, answer_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool2")
                answer_outputs.append(pooled2)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)

        self.question_pool = tf.concat(question_outputs, 2)
        self.question_pool_flat = tf.reshape(self.question_pool, [-1, num_filters_total])

        self.answer_pool = tf.concat(answer_outputs, 2)
        self.answer_pool_flat = tf.reshape(self.answer_pool, [-1, num_filters_total])

        with tf.name_scope("full-connected"):
            W = tf.Variable(tf.truncated_normal([num_filters_total, 64], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[64]), name="b")
            self.question_outputs = tf.matmul(self.question_pool_flat, W) + b
            self.answer_outputs = tf.matmul(self.answer_pool_flat, W) + b
        with tf.name_scope("loss"):
            # self.labels=tf.reshape(self.labels,[-1, 32])
            self.loss=tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=
                        tf.reshape(
                            tf.diag_part(
                                tf.matmul(
                                    tf.transpose(self.question_outputs), self.answer_outputs)),
                                [-1, 1]), 
                labels=self.labels))
        with tf.name_scope("accuracy"):
            self.a = tf.round(tf.sigmoid(tf.reshape(
                            tf.diag_part(
                                tf.matmul(
                                    tf.transpose(self.question_outputs), self.answer_outputs)),
                                [-1, 1])), name=None)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.a, self.labels), tf.float32))

