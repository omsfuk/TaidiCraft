# -*- encoding: utf-8 -*-
import tensorflow as tf
import numpy as np
#from sklearn.metrics import roc_auc_score

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, vocab_size, question_length, answer_length,
      embedding_size, classes_num, num_filters, filter_sizes,l2_reg_lambda=0.0, embeddingW=None):
        # Placeholders for input, output and dropout

        self.questions = tf.placeholder(tf.int32, [None, question_length], name="questions")
        self.answers = tf.placeholder(tf.int32, [None, answer_length], name="answers")
        self.labels = tf.placeholder(tf.float32, [None, classes_num], name="labels")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        l2_loss = tf.constant(0.0)
        # Embedding layer
        with tf.device('/cpu:0'),tf.name_scope("Embedding"):
            if embeddingW is not None:
                self.W = tf.cast(tf.Variable(embeddingW, name="W"), tf.float32)
            else:
                self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    trainable=True, name="W")
            '''self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")    '''            

            self.questionEmbedding1 = tf.nn.embedding_lookup(self.W, self.questions)
            self.questionEmbedding = tf.reshape(self.questionEmbedding1, [-1, question_length, embedding_size])
            self.questionEmbedding_expanded = tf.expand_dims(self.questionEmbedding, -1)

            self.answerEmbedding1 = tf.nn.embedding_lookup(self.W, self.answers)
            self.answerEmbedding = tf.reshape(self.answerEmbedding1, [-1, answer_length, embedding_size])
            self.answerEmbedding_expanded = tf.expand_dims(self.answerEmbedding, -1)

        # Create a convolution + maxpool layer for each filter size
        question_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with  tf.name_scope("question-conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W2= tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W1")
                b2 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b1")
                conv2 = tf.nn.conv2d(
                    self.questionEmbedding_expanded,
                    W2,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv2")
                # Apply nonlinearity
                h2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), name="relu2")
                # Maxpooling over the outputs
                pooled2 = tf.nn.max_pool(
                    h2,
                    ksize=[1, question_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool2")
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
                h2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), name="relu2")
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

        self.question_pool = tf.concat(question_outputs, 3)
        self.question_pool_flat = tf.reshape(self.question_pool, [-1, num_filters_total])

        self.answer_pool = tf.concat(answer_outputs, 3)
        self.answer_pool_flat = tf.reshape(self.answer_pool, [-1, num_filters_total])
		
        #self.text_pool_flat = tf.concat((question_pool_flat, answer_pool_flat),axis=1)

        def cosine(x3, x4):
            x3_norm = tf.sqrt(tf.reduce_sum(tf.square(x3), axis=1))
            x4_norm = tf.sqrt(tf.reduce_sum(tf.square(x4), axis=1))
            #内积
            x3_x4 = tf.reduce_sum(tf.multiply(x3, x4), axis=1)
            return tf.divide(x3_x4, tf.multiply(x3_norm, x4_norm))
        # Add dropout
        with tf.name_scope("dropout"):
            self.question_drop = tf.nn.dropout(self.question_pool_flat, self.dropout_keep_prob)
            self.answer_drop = tf.nn.dropout(self.answer_pool_flat, self.dropout_keep_prob)
        with tf.name_scope("full-connected"):
            W3 = tf.Variable(tf.truncated_normal([num_filters_total, classes_num], stddev=0.1), name="W3")
            b3 = tf.Variable(tf.constant(0.1, shape=[classes_num]), name="b3")
            l2_loss += tf.nn.l2_loss(W3)
            l2_loss += tf.nn.l2_loss(b3)
            self.question_outputs = tf.matmul(self.question_drop, W3, name="question_outputs") + b3
            self.answer_outputs = tf.matmul(self.answer_drop, W3, name="answer_outputs") + b3
            self.scores = tf.reshape(cosine(self.question_outputs, self.answer_outputs), [-1, 1])
        with tf.name_scope("loss"):
            self.loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=tf.cast(tf.reshape(tf.argmax(self.labels, 1),[-1,1]),tf.float32)))+l2_reg_lambda * l2_loss
        with tf.name_scope("accuracy"):
            self.a = tf.round(tf.sigmoid(self.scores), name="predictions")
            correct_predictions = tf.equal(self.a, tf.cast(tf.reshape(tf.argmax(self.labels,1),[-1,1]),tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions,tf.float32))
