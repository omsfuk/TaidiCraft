# -*- encoding: utf-8 -*-
import tensorflow as tf
import json
import jieba
import numpy as np
import re
import time
import os
import datetime
import gensim
from text_cnn_craft import TextCNN
from lib_craft import _now
from lib_craft import expand_array 

# 正文匹配，过滤特殊字符
pattern = re.compile(r'[\u4e00-\u9fa5_a-zA-Z0-9１２３４５６７８９０]')
# 索引词典
dic = {"$$zero$$": 0}
# word2vec model
print("[%s] loading word2vec model..." % _now())
model = gensim.models.Word2Vec.load('npy/word2vec_wx')
# 过滤特殊符号
punct = set(u'''#ㄍ <>/\\[]:!)］∫,.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒
        ﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠
        々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻
        ︽︿﹁﹃﹙﹛﹝（｛“‘-—_…... ''')
filter_punt = lambda s: u''.join(filter(lambda x: True if pattern.match(x) and x not in punct else False , s))
embeddingW = []
# 有效数据条数
valid_sample = 0

# 常量定义
tf.flags.DEFINE_integer("batch_size", 64, "数据集大小")
tf.flags.DEFINE_integer("epoch_num", 20, "迭代次数")
tf.flags.DEFINE_integer("max_question_length", 50, "最大问题长度")
tf.flags.DEFINE_integer("min_question_length", 2, "最小问题长度")
tf.flags.DEFINE_integer("max_answer_length", 64, "最大答案长度")
tf.flags.DEFINE_integer("min_answer_length", 5, "最小答案长度")
tf.flags.DEFINE_integer("embedding_size", 256, "embedding size")
tf.flags.DEFINE_string("filter_sizes", "3, 4, 5", "filter sizes")
tf.flags.DEFINE_integer("filter_num", 128, "filternum")
tf.flags.DEFINE_float("dev_sample_percentage", 0.2, "测试集比例")
tf.flags.DEFINE_integer("evaluate_every", 100, "两次评估间隔")
tf.flags.DEFINE_integer("word_precess_every", 5000, "单词处理信息打印间隔")
tf.flags.DEFINE_integer("used_sample", 6400, "限制样本实际利用大小。当为None时为无限制")
tf.flags.DEFINE_integer("num_checkpoints", 5, "检查点数量")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

"""
获得样本大小
"""
def get_sample_size():
    with open('train_data_sample.json', 'r', encoding='utf-8') as f:
        json_obj = json.load(f)
    return len([1 for qa in json_obj for ans in qa['passages']])



"""
词序列到向量
dest_length: 目标向量行数
"""
def convert_to_word_vector(senquence, dest_length):
    assert len(senquence) <= dest_length
    ans = []
    for word in senquence:
        if word in dic.keys():
            ans.append(dic[word])
        else:
            index = len(dic)
            dic[word] = index
            if word not in model.wv.vocab:
                vector = np.random.rand(FLAGS.embedding_size)
            else:
                vector = np.array(model[word])
            embeddingW.append(vector)
            ans.append(index)
    ans = expand_array(ans, dest_length=dest_length)
    return np.array(ans)

"""
生成训练数据。分批生成，节约Menory。假定样本书不超过1 * 10^^8
"""
def batch_iter(batch_size, epoch_num, start_pos=0, end_pos=100000000, get_valid_sample=False):
    global valid_sample
    res = []
    with open('train_data_sample.json', 'r',encoding='utf-8') as f:
        json_obj = json.load(f)
    for epoch_count in range(0, epoch_num):
        line_count = 0
        if get_valid_sample == False:
            print("[%s] epoch %d" % (datetime.datetime.now().strftime('%b-%d-%y %H:%M:%S'), epoch_count + 1))
        for qa in json_obj:
            if line_count > end_pos:
                break
            question_seg = jieba.lcut(filter_punt(qa['question']), cut_all=False) # 问题 词序列

            # 问题长度过滤
            if len(question_seg) > FLAGS.max_question_length or len(question_seg) < FLAGS.min_question_length:
                continue

            for ans in qa['passages']:
                # 样本数量限制
                line_count = line_count + 1
                if line_count > end_pos:
                    break

                if line_count > start_pos:
                    answer_seg = jieba.lcut(filter_punt(ans['content']), cut_all=False) # 答案 词序列
                    # 答案长度过滤
                    if len(answer_seg) > FLAGS.max_answer_length or len(answer_seg) < FLAGS.min_question_length:
                        continue

                    vector = (np.array((ans['label'] * 1.0, )).tolist(),
                                    convert_to_word_vector(question_seg, FLAGS.max_question_length).tolist(),
                                    convert_to_word_vector(answer_seg, FLAGS.max_answer_length).tolist()) # 一行向量 (label, question, answer)
                    res.append(vector)

                    # 获取有效数据 模式
                    if get_valid_sample == True:
                        valid_sample = valid_sample + 1
                        if line_count % FLAGS.word_precess_every == 0:
                            print("[%s] processing %d question/answer pair" % (_now(), line_count))
                    else:
                        if len(res) == batch_size:
                            res = np.array(res)
                            shuffle_indices = np.random.permutation(np.arange(batch_size))
                            shuffled_data = res[shuffle_indices]
                            yield shuffled_data.tolist()
                            res = []

print("[%s] getting extract statistics..." % _now())
embeddingW.append(np.zeros((FLAGS.embedding_size)))
sample_size = dev_end_pos = get_sample_size() if FLAGS.used_sample is None else FLAGS.used_sample
train_end_pos = dev_start_pos = (sample_size * (1 - FLAGS.dev_sample_percentage))
for _ in batch_iter(FLAGS.batch_size, 1, 0, sample_size, get_valid_sample=True):
    continue
print("total_sample:\t\t %d" % sample_size)
print("valid_sample:t\t %d" % valid_sample)
print("dict_size: %d\t\t" % len(dic))
print("train_sample:\t\t [%d, %d]" % (0, train_end_pos))
print("dev_sample:\t\t [%d, %d]" % (dev_start_pos, dev_end_pos))

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(vocab_size=len(dic),
                    question_length=FLAGS.max_question_length,
                    answer_length=FLAGS.max_answer_length,
                    embedding_size=FLAGS.embedding_size,
                    batch_size=FLAGS.batch_size,
                    num_filters=FLAGS.filter_num,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
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
        
        # Dev Summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(labels, questions, answers):
            # A single training step
            feed_dict = {
              cnn.questions: questions,
              cnn.answers: answers,
              cnn.labels: labels,
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict)
                                    
            time_str = datetime.datetime.now().isoformat()                
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)
     
        def dev_step(labels, questions, answers):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.questions: questions,
                cnn.answers: answers,
                cnn.labels: labels
            }
            step, summaries, loss, accuracy = sess.run([global_step, dev_summary_op, cnn.loss, cnn.accuracy], feed_dict)
            dev_summary_writer.add_summary(summaries, step)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            return (loss, accuracy)

        # Generate batches
        batches = batch_iter(FLAGS.batch_size, FLAGS.epoch_num, start_pos=0, end_pos=train_end_pos)
        # Training loop. For each batch...
        for batch in batches:
            labels, questions, answers = zip(*batch)
            train_step(labels=np.array(labels), questions=np.array(questions), answers=np.array(answers))
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_batchs= batch_iter(FLAGS.batch_size, 1, start_pos=dev_start_pos, end_pos=dev_end_pos)
                ans = []
                for dev_batch in dev_batchs:
                    labels, questions, answers = zip(*dev_batch)
                    ans.append(dev_step(labels=np.array(labels), questions=np.array(questions), answers=np.array(answers)))
                ans = np.average(ans, axis=0)
                print("loss {:g}, acc {:g}".format(ans.tolist()[0], ans.tolist()[1]))
                print("")

