# -*- encoding: utf-8 -*-
import tensorflow as tf
import numpy as np
import time
import datetime
import os
import gensim
import pickle
from text_cnn_craft import TextCNN
from lib_craft import mprint

train_file = "train_data_complete_10000_2_2.vec"
test_file = "testing.vec"
batch_size = 64
epoch_num = 100
embedding_size = 256

filter_sizes = "3, 4, 5"
filter_num = 128
dropout_keep_prob = 0.5
l2_reg_lambda = 0.5

evaluate_every = 5
checkpoint_num = 5
checkpoint_every = 5

question_length = 16
answer_length = 32


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

def get_sample(train_file, test_file):
    mprint("Load vector ...")
    with open(train_file, "rb") as f:
        data_train = pickle.load(f)
        data_train = np.delete(data_train, 0, 1)

    with open(test_file, "rb") as f:
        data_dev = pickle.load(f)
        data_train = np.delete(data_train, 0, 1)
    mprint("Complete")
    return (data_train, data_dev)

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
        f.write("dropout \t\t{}\n".format(dropout_keep_prob))
        f.write("l2      \t\t{}\n".format(l2_reg_lambda))
        f.write("filter_num\t\t{}\n".format(filter_num))
        f.write("filter_size\t\t{}\n".format(filter_sizes))

def load_model():
    mprint("Loading word2vec model ...")
    model = gensim.models.Word2Vec.load('npy/word2vec_wx')
    mprint("Complete.")
    return model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
data_train, data_dev = get_sample(train_file, test_file)

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
                    l2_reg_lambda=l2_reg_lambda,
                    filter_sizes=list(map(int, filter_sizes.split(","))),
                    embeddingW=np.array(embeddingW).astype(np.float32))

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-4)
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

        def train_step(epoch, labels, questions, answers):
            # A single training step
            feed_dict = {
              cnn.questions: questions,
              cnn.answers: answers,
              cnn.labels: labels,
              cnn.dropout_keep_prob: dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict)
                                    
            mprint("epoch {}, step {}, loss {:g}, acc {:g}".format(epoch, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)
     
        def dev_step(labels, questions, answers):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.questions: questions,
                cnn.answers: answers,
                cnn.labels: labels,
                cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy], feed_dict)

            # dev_summary_writer.add_summary(summaries, step)
            mprint("step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
            return (loss, accuracy)

        # Generate batches
        batches = batch_iter(data_train, batch_size, epoch_num)
        # Training loop. For each batch...
        for epoch, batch in batches:
            labels, questions, answers = zip(*batch)
            train_step(epoch=epoch, labels=np.array(labels), questions=np.array(questions), answers=np.array(answers))
            current_step = tf.train.global_step(sess, global_step)
            if current_step % evaluate_every == 0:
                print("\nEvaluation:")
                summary = tf.Summary()
                dev_batchs = batch_iter(data_dev, batch_size, 1)
                ans = []
                for epoch, dev_batch in dev_batchs:
                    labels, questions, answers = zip(*dev_batch)
                    ans.append(dev_step(labels=np.array(labels), questions=np.array(questions), answers=np.array(answers)))
                avg_loss, avg_acc = np.average(ans, axis=0).tolist()
                summary.value.add(tag="loss", simple_value=avg_loss)
                summary.value.add(tag="Accuracy", simple_value=avg_acc)
                dev_summary_writer.add_summary(summary, current_step)
                mprint("Summary:   loss {:g}, acc {:g}\n".format(avg_loss, avg_acc))
            if current_step % checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                mprint("Saved model checkpoint to {}\n".format(path))










