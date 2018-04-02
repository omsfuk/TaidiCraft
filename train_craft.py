# -*- encoding: utf-8 -*-
import tensorflow as tf
import pickle
import json
import jieba
import numpy as np
import re
import time
import os
import datetime
import gensim
from text_cnn_craft import TextCNN

input = open("train_data_sample.json",encoding='utf-8')
pattern = re.compile(r'[\u4e00-\u9fa5_a-zA-Z0-9１２３４５６７８９０]')
# dic = pickle.load(open("word.index", 'rb'))
# dic['default_word'] = 0
dic = {}
batch_size = 64
num_epochs = 5
question_length = 50
answer_length = 64
embedding_size = 128
num_filters = 128
filter_sizes = (5, 6, 7) 
dev_sample_percentage = 0.1
embeddingW = []

model = gensim.models.Word2Vec.load('npy/word2vec_wx')

tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    print("valid data %d" % len(data))
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        print("epoch %d" % epoch)
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def to_vector(senquence):
    ans_list = ()
    for word in senquence:
        if word in dic.keys():
            ans_list = ans_list + (dic[word], )
        else:
            index = len(dic)
            dic[word] = index
            if word not in model.wv.vocab:
                embeddingW.append(np.random.rand(1, embedding_size))
            else:
                embeddingW.append(model[word])
            ans_list = ans_list + (index, )

    return np.array(ans_list)

# 过滤特殊符号
punct = set(u'''#ㄍ <>/\\[]:!)］∫,.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒
        ﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠
        々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻
        ︽︿﹁﹃﹙﹛﹝（｛“‘-—_…''')
# 对str/unicode
# filter_punt = lambda s: u''.join(filter(lambda x: x not in punct, s))
filter_punt = lambda s: u''.join(filter(lambda x: True if pattern.match(x) else False , s))

def fill_list(elements, value, size):
    if len(elements) < size:
        for _ in range(0, size - len(elements)):
            elements.append(value)
    return elements

def preprocess():
    data = []
    text_data = json.loads(input.read())
    ans_index = 0
    line_limit = 0
    for qa in text_data:
        if line_limit > 6400:
            break
        question_seg = jieba.lcut(filter_punt(qa['question']), cut_all=False)
        question_seg = fill_list(question_seg, "default_word", question_length)
        answers = qa['passages']
        for ans in answers:
            line_limit = line_limit + 1
            if line_limit > 6400:
                break
            ans_index = ans_index + 1
            print("try to process " + str(ans_index) + "question/answer")
            answer_seg = jieba.lcut(filter_punt(ans['content']), cut_all=False)
            if len(answer_seg) <= answer_length:
                answer_seg = fill_list(answer_seg, "default_word", answer_length)
                # line_vector = np.hstack((np.array((ans['label'] * 1.0, )), to_vector(question_seg + answer_seg)))
                line_vector = (np.array((ans['label'] * 1.0, )), to_vector(question_seg), to_vector(answer_seg))
                data.append(line_vector)

    input.close()
    return data





def train():
    data = np.array(preprocess())
    print(len(embeddingW))

    # Randomly shuffle data
    np.random.seed(10)

    shuffle_indices = np.random.permutation(np.arange(len((data))))
    data_shuffled = data[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(dev_sample_percentage * float(len(data)))
    data_train, data_dev = data_shuffled[:dev_sample_index], data_shuffled[dev_sample_index:]



    FLAGS = tf.flags.FLAGS
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(vocab_size=len(model.wv.vocab),
                        question_num=64,
                        answer_num=64,
                        question_length=question_length,
                        answer_length=answer_length,
                        embedding_size=embedding_size,
                        batch_size=batch_size,
                        num_filters=num_filters,
                        filter_sizes=(4, 4, 5)
                        )

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            # vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(questions, answers, labels):
                # A single training step
                feed_dict = {
                  cnn.questions: questions,
                  cnn.answers: answers,
                  cnn.labels: labels,
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, labels, writer=None):
                # Evaluates model on a dev set
                feed_dict = {
                  cnn.questions: x_batch,
                  cnn.answers: y_batch,
                  cnn.labels: labels,
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = batch_iter(data_train[0:int(len(data_train) / batch_size) * batch_size], batch_size, num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                labels = []
                questions = []
                answers = []
                for item in batch:
                    label, question, answer = item
                    labels.append(label)
                    questions.append(question)
                    answers.append(answer)
                train_step(questions=questions, answers=answers, labels=labels)
                current_step = tf.train.global_step(sess, global_step)
                """
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    labels = []
                    questions = []
                    answers = []
                    data_dev = data_dev[0:int(len(data_dev) / batch_size) * batch_size]
                    for item in data_dev:
                        label, question, answer = item
                        labels.append(label)
                        questions.append(question)
                        answers.append(answer)

                    dev_step(questions, answers, labels, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                """

train()
dic_out = open("word.index", "wb")
pickle.dump(dic, dic_out)
dic_out.close()
