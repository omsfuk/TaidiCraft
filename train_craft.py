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
question_outputs=[]
answer_outputs=[]
dic = {'default_word': 0}
batch_size = 64
num_epochs = 5
question_length = 50
answer_length = 64
embedding_size = 256
num_filters = 128
filter_sizes = (5, 6, 7) 
dev_sample_percentage = 0.1
embeddingW = []
embeddingW.append(np.zeros(shape=(embedding_size)))
sample_limit = 640

model = gensim.models.Word2Vec.load('npy/word2vec_wx')

tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")

# 过滤特殊符号
punct = set(u'''#ㄍ <>/\\[]:!)］∫,.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒
        ﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠
        々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻
        ︽︿﹁﹃﹙﹛﹝（｛“‘-—_…... ''')
# 对str/unicode
# filter_punt = lambda s: u''.join(filter(lambda x: x not in punct, s))
filter_punt = lambda s: u''.join(filter(lambda x: True if pattern.match(x) else False , s))

def fill_list(elements, value, size):
    if len(elements) < size:
        for _ in range(0, size - len(elements)):
            elements.append(value)
    return elements

def to_vector(senquence):
    ans_list = ()
    for word in senquence:
        if word in dic.keys():
            ans_list = ans_list + (dic[word], )
        else:
            index = len(dic)
            dic[word] = index
            if word not in model.wv.vocab:
                embeddingW.append(np.random.rand(embedding_size))
            else:
                embeddingW.append(np.array(model[word]))
            ans_list = ans_list + (index, )

    return np.array(ans_list)


def batch_iter(batch_size, epoch_num):
    res = []
    with open('train_data_sample.json', 'r',encoding='utf-8') as f:
        json_obj = json.load(f)
    line_limit = 0
    ans_index = 0
    for epoch_count in range(0, epoch_num):
        print("[%s] epoch %d" % (datetime.datetime.now().strftime('%b-%d-%y %H:%M:%S'), epoch_count + 1))
        for qa in json_obj:
            if line_limit > sample_limit:
                break
            question_seg = jieba.lcut(filter_punt(qa['question']), cut_all=False) # 问题 词序列

            # 问题长度过滤
            if len(question_seg) > question_length:
                continue

            question_seg = fill_list(question_seg, "default_word", question_length)
            answers = qa['passages'] 
            for ans in answers:
                # 样本数量限制
                line_limit = line_limit + 1
                if line_limit > sample_limit:
                    break

                ans_index = ans_index + 1
                if ans_index % 100 == 0:
                    print("processing %d question and answer" % ans_index)

                answer_seg = jieba.lcut(filter_punt(ans['content']), cut_all=False) # 答案 词序列
                # 答案长度过滤
                if len(answer_seg) > answer_length:
                    continue

                answer_seg = fill_list(answer_seg, "default_word", answer_length)
                line_vector = (np.array((ans['label'] * 1.0, )),
                        to_vector(question_seg),
                        to_vector(answer_seg)) # 一行向量 (label, question, answer)
                res.append(line_vector)
                if len(res) == batch_size:
                    res = np.array(res)
                    shuffle_indices = np.random.permutation(np.arange(batch_size))
                    shuffled_data = res[shuffle_indices]
                    yield shuffled_data
                    res = []

def train():

    for _ in batch_iter(batch_size, 1):
        continue

    FLAGS = tf.flags.FLAGS
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(vocab_size=len(dic),
                        question_length=question_length,
                        answer_length=answer_length,
                        embedding_size=embedding_size,
                        batch_size=batch_size,
                        num_filters=num_filters,
                        filter_sizes=(2, 3, 4),
                        embeddingW=np.array(embeddingW))

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

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(questions, answers, labels):
                # A single training step
                feed_dict = {
                  cnn.questions: questions,
                  cnn.answers: answers,
                  cnn.labels: labels,
                }
                """
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                """
                global question_outputs,answer_outputs
                _, step, summaries, loss, accuracy, question_outputs, answer_outputs= sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.question_outputs, cnn.answer_outputs],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print(len(question_outputs[0]))
                print(len(answer_outputs[0]))
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

         #定义了一个函数，用于验证集，输入为一个batch
            '''def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                #验证集太大，会爆内存，采用batch的思想进行计算，下面生成多个子验证集
                num=20
                x_batch=x_batch.tolist()
                y_batch=y_batch.tolist()
                l=len(y_batch)
                l_20=int(l/num)
                x_set=[]
                y_set=[]
                for i in range(num-1):
                    x_temp=x_batch[i*l_20:(i+1)*l_20]
                    x_set.append(x_temp)
                    y_temp=y_batch[i*l_20:(i+1)*l_20]
                    y_set.append(y_temp)
                x_temp=x_batch[(num-1)*l_20:]
                x_set.append(x_temp)
                y_temp=y_batch[(num-1)*l_20:]
                y_set.append(y_temp)

                #每个batch验证集计算一下准确率，num个batch再平均
                lis_loss=[]
                lis_accu=[]
                for i in range(num):    
                    feed_dict = {
                    cnn.input_x: np.array(x_set[i]),
                    cnn.input_y: np.array(y_set[i]),
                    #cnn.dropout_keep_prob: 1.0
                    }
                    step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                    lis_loss.append(loss)
                    lis_accu.append(accuracy)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    print("test_loss and test_acc"+"\t\t"+str(sum(lis_loss)/num)+"\t\t"+str(sum(lis_accu)/num))
                    if writer:
                        writer.add_summary(summaries, step)'''

            # Generate batches
            batches = batch_iter(batch_size, num_epochs)
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
                '''if current_step % FLAGS.evaluate_every == 0:#每多少步，算一下验证集效果
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)#喂的数据为验证集，此时大小不止一个batchsize1的大小
                print("")'''
train()
dic_out = open("word.index", "wb")
pickle.dump(dic, dic_out)
dic_out.close()
