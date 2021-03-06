#!/usr/bin/python3
# -*- encoding: utf-8 -*-
import tensorflow as tf
import numpy as np
import time
import datetime
import os
import gensim
import pickle
from text_cnn_craft_onemodel import TextCNN
from lib_craft import mprint

oo = 2147483647
train_file = "training_320_1_1.vec"
test_file = "testing_320_1_1.vec"
test_file2 = "testing_320_1_1.vec"
test_file3 = "testing_320_1_1.vec"
batch_size = 128
epoch_num = oo
embedding_size = 256
learn_rate = 0.0001

filter_sizes = "1, 3, 5"
filter_num = 128
dropout_keep_prob = 0.01
l2_reg_lambda = 0.6
CUDA_VISIBLE_DEVICE = "0"

evaluate_every = 50
checkpoint_num = 3
checkpoint_every = 50

question_length = 50
answer_length = 256

feature_size = 10


"""
生成训练数据
"""
def batch_iter(data, batch_size, epoch_num, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(1, epoch_num + 1):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield (epoch, shuffled_data[start_index:end_index])

def get_sample(train_file, test_file, test_file2, test_file3):
    mprint("Load vector ...")
    with open(train_file, "rb") as f:
        data_train = pickle.load(f)

    with open(test_file, "rb") as f:
        data_dev = pickle.load(f)
    with open(test_file2, "rb") as f:
        data_dev2 = pickle.load(f)
    with open(test_file3, "rb") as f:
        data_dev3 = pickle.load(f)

    mprint("Complete")
    return (data_train, data_dev, data_dev2, data_dev3)

"""
构建embeddingW词典
"""
def construct_embeddingW(dic):
    mprint("Contructing embeddingW ...")
    ordered_dic = sorted(dic.items(), key=lambda x: x[1])
    ans = []
    for item in ordered_dic:
        word = item[0]
        if word in model.wv.vocab:
            ans.append(model[word])
        else:
            raise SystemExit("人为设置的终止，这不可能")
            # ans.append(np.random.rand(embedding_size))
    mprint("Complete.")
    return ans

def write_parameter_file():
    global out_dir
    timestamp = str(int(time.time()))
    os.makedirs(os.path.join(os.path.curdir, "runs", timestamp))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
 
    with open(os.path.join(out_dir, "parameter.txt"), "w", encoding="utf-8") as f:
        f.write("train _ file \t\t{}\n".format(train_file))
        f.write("testing_file \t\t{}\n".format(test_file))
        f.write("dropout \t\t{}\n".format(dropout_keep_prob))
        f.write("l2      \t\t{}\n".format(l2_reg_lambda))
        f.write("filter_num\t\t{}\n".format(filter_num))
        f.write("filter_size\t\t{}\n".format(filter_sizes))
        f.write("batch_size\t\t{}\n".format(batch_size))
        f.write("question_length\t\t{}\n".format(question_length))
        f.write("answer_length\t\t{}\n".format(answer_length))
        f.write("learning_rate\t\t{}\n".format(learn_rate))
def load_model():
    mprint("Loading word2vec model ...")
    model = gensim.models.Word2Vec.load('npy/word2vec_wx')
    mprint("Complete.")
    return model

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICE
data_train, data_dev, data_dev2, data_dev3 = get_sample(train_file, test_file, test_file2, test_file3)

# shuffle data_dev
data_dev = np.array(data_dev)
shuffle_indices = np.random.permutation(np.arange(len(data_dev)))
data_dev = data_dev[shuffle_indices]

data_dev2 = np.array(data_dev2)
shuffle_indices = np.random.permutation(np.arange(len(data_dev2)))
data_dev2 = data_dev2[shuffle_indices]

data_dev3 = np.array(data_dev3)
shuffle_indices = np.random.permutation(np.arange(len(data_dev3)))
data_dev3 = data_dev3[shuffle_indices]

del shuffle_indices
 
# 索引词典
if os.path.isfile("word.index"):
    with open("word.index", "rb") as f:
        dic = pickle.load(f)
else:
    raise SystemExit("Can't find word.index")

model = load_model()
embeddingW = construct_embeddingW(dic)

write_parameter_file()

with tf.Graph().as_default():
    session_conf = tf.ConfigProto()
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(vocab_size=len(dic),
                    question_length=question_length,
                    answer_length=answer_length,
                    embedding_size=embedding_size,
                    num_filters=filter_num,
                    classes_num=2,
                    feature_size=feature_size,
                    l2_reg_lambda=l2_reg_lambda,
                    filter_sizes=list(map(int, filter_sizes.split(","))),
                    embeddingW=np.array(embeddingW).astype(np.float32))

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learn_rate)
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
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        
        # Dev Summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=checkpoint_num)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        # sess.run(tf.local_variables_initializer()) # auc needed

        def train_step(epoch, labels, questions, answers, question_feature, answer_feature):
            # A single training step
            avg_acc = []
            avg_loss = []
            feed_dict = {
              cnn.questions: questions,
              cnn.answers: answers,
              cnn.labels: labels,
              cnn.dropout_keep_prob: dropout_keep_prob,
              cnn.question_feature: question_feature,
              cnn.answer_feature: answer_feature
            }
            _, step, loss, accuracy = sess.run(
               [train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)
            avg_acc.append(accuracy)
            avg_loss.append(loss)
                                    
            mprint("epoch {}, step {}, loss {:g}, acc {:g}".format(epoch, step, loss, accuracy))
            # train_summary_writer.add_summary(summaries, step)
            
            return (np.average(avg_loss), np.average(avg_acc))
     
        def dev_step(labels, questions, answers, question_feature, answer_feature):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.questions: questions,
                cnn.answers: answers,
                cnn.labels: labels,
                cnn.dropout_keep_prob: 1.0,
                cnn.question_feature: question_feature,
                cnn.answer_feature: answer_feature
            }
            step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy], feed_dict)

            dev_summary_writer.add_summary(summaries, step)
            print("")
            mprint("Evaluation: step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
            return (loss, accuracy)

        max_acc = 0
        # Generate batches
        batches = batch_iter(data_train, batch_size, epoch_num)
        # Training loop. For each batch...
        for epoch, batch in batches:
            _, labels, questions, answers, q_feature, a_feature = zip(*batch)
            train_step(epoch=epoch, labels=np.array(labels), questions=np.array(questions), answers=np.array(answers), question_feature=q_feature, answer_feature=a_feature)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % evaluate_every == 0:
                # print("\nEvaluation:")
                _, labels, questions, answers, q_feature, a_feature = zip(*data_dev)
                loss, acc = dev_step(labels=np.array(labels), questions=np.array(questions), answers=np.array(answers), question_feature=q_feature, answer_feature=a_feature)
                
                _, labels, questions, answers, q_feature, a_feature = zip(*data_dev2)
                loss, acc = dev_step(labels=np.array(labels), questions=np.array(questions), answers=np.array(answers), question_feature=q_feature, answer_feature=a_feature)

                _, labels, questions, answers, q_feature, a_feature = zip(*data_dev2)
                loss, acc = dev_step(labels=np.array(labels), questions=np.array(questions), answers=np.array(answers), question_feature=q_feature, answer_feature=a_feature)

                """
                summary = tf.Summary()
                dev_batchs = batch_iter(data_dev, batch_size, 1)
                ans = []
                for epoch, dev_batch in dev_batchs:
                    _, labels, questions, answers, q_feature, a_feature = zip(*dev_batch)
                    ans.append(dev_step(labels=np.array(labels), questions=np.array(questions), answers=np.array(answers), question_feature=q_feature, answer_feature=a_feature))
                avg_loss, avg_acc = np.average(ans, axis=0).tolist()
                summary.value.add(tag="loss", simple_value=avg_loss)
                summary.value.add(tag="Accuracy", simple_value=avg_acc)
                dev_summary_writer.add_summary(summary, current_step)
                mprint("Summary:   step {} loss {:g}, acc {:g}\n".format(current_step, avg_loss, avg_acc))
                """
            if current_step % checkpoint_every == 0 and acc > max_acc:
                max_acc = acc
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                mprint("Saved model checkpoint to {}\n".format(path))

