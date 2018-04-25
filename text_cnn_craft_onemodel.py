# -*- encoding: utf-8 -*-
import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, vocab_size, question_length, answer_length, feature_size,
      embedding_size, classes_num, num_filters, filter_sizes, l2_reg_lambda, embeddingW=None):
        # Placeholders for input, output and dropout

        self.questions = tf.placeholder(tf.int32, [None, question_length], name="questions")
        self.answers = tf.placeholder(tf.int32, [None, answer_length], name="answers")
        self.labels = tf.placeholder(tf.float32, [None, classes_num], name="labels")
        self.question_feature = tf.placeholder(tf.float32, [None, feature_size], name="question_feature")
        self.answer_feature = tf.placeholder(tf.float32, [None, feature_size], name="answer_feature")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

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
                W2 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W2")
                b2 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b2")
                conv2 = tf.nn.conv2d(
                    self.textEmbedding_expanded,
                    W2,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv2")
                # Apply nonlinearity
                h2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), name="relu")
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
        self.text_pool_flat = tf.concat([self.text_pool_flat, self.question_feature, self.answer_feature], 1)

        def cosine(x3, x4):
            x3_norm = tf.sqrt(tf.reduce_sum(tf.square(x3), axis=1))
            x4_norm = tf.sqrt(tf.reduce_sum(tf.square(x4), axis=1))
            #内积
            x3_x4 = tf.reduce_sum(tf.multiply(x3, x4), axis=1)
            return tf.divide(x3_x4, tf.multiply(x3_norm, x4_norm))

        with tf.name_scope("dropout"):
            self.text_drop = tf.nn.dropout(self.text_pool_flat, self.dropout_keep_prob)

        with tf.name_scope("output"):
            # W = tf.Variable(tf.truncated_normal([num_filters_total + feature_size * 2, classes_num], stddev=0.1), name="W")
            W = tf.get_variable(
                "W",
                shape=[num_filters_total + feature_size * 2, classes_num],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[classes_num]), name="b")
            h3 = tf.nn.relu(tf.nn.bias_add(W, b))
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.text_outputs = tf.matmul(self.text_pool_flat, W) + b
        with tf.name_scope("loss"):
            self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.text_outputs, labels=self.labels)) + l2_reg_lambda * l2_loss
        with tf.name_scope("accuracy"):
            self.predictions = tf.argmax(self.text_outputs, 1, "predictions")
            self.precision = tf.divide(tf.reduce_sum(tf.argmax(self.text_outputs, 1, "precision")), tf.cast(tf.shape(self.labels)[0], tf.int64))
            self.recall = tf.divide(tf.reduce_sum(tf.argmax(self.text_outputs, 1, "recall")), tf.reduce_sum(tf.argmax(self.labels, 1, "recall")))
            # self.f1 = tf.divide(tf.multiply(tf.multiply(self.precision, self.recall), 2), self.precision + self.recall, name="f1")
            self.f1 = tf.divide(tf.multiply(tf.reduce_sum(tf.multiply(tf.argmax(self.text_outputs, 1), tf.argmax(self.labels, 1))), 2), 
                    tf.reduce_sum(tf.argmax(self.text_outputs, 1)) + tf.reduce_sum(tf.argmax(self.labels, 1)), name="f1")

            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, tf.argmax(self.labels, 1)), tf.float32))
            # self.accuracy = tf.metrics.auc(self.predictions, tf.argmax(self.labels, 1))[1]
            self.eval_accuracy = tf.metrics.auc(self.predictions, tf.argmax(self.labels, 1))[1]

