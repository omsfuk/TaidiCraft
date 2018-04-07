#! /usr/bin/env python
# -*- encoding:utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import pickle
import json
import jieba
import re
import csv
import math
from lib_craft import expand_array
from lib_craft import _now
from tensorflow.contrib import learn

p = re.compile('[\u4e00-\u9fa50-9a-zA-Z]+')
all_predictions = []
# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("evaluate_file", "unbalance_testing_4000.json", "最小答案长度")
tf.flags.DEFINE_integer("max_question_length", 50, "最大问题长度")
tf.flags.DEFINE_integer("min_question_length", 2, "最小问题长度")
tf.flags.DEFINE_integer("max_answer_length", 64, "最大答案长度")
tf.flags.DEFINE_integer("min_answer_length", 5, "最小答案长度")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")
tf.flags.DEFINE_string("checkpoint_dir", "runs/1523104641/checkpoints", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS

with open(os.path.join(os.path.curdir, FLAGS.checkpoint_dir, "..", "word.index"), "rb") as f:
    dic = pickle.load(f)
"""
punct = set(u'''#ㄍ <>/\\[]:!)］∫,.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒
        ﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠
        々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻
        ︽︿﹁﹃﹙﹛﹝（｛“‘-—_…... ''')
"""
#filter_punt = lambda s: u''.join(filter(lambda x: True if pattern.match(x) and x not in punct else False , s))
filter_punt = lambda s: p.match(s)

# 分词
def cut_cut(seq):
    return [x for x in filter(filter_punt, jieba.lcut(seq, cut_all=False))]

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
    ans = expand_array(ans, dest_length=dest_length)
    return np.array(ans)
 
"""
初始化sample && 词向量。json => list
"""
def init(end_pos, max_question_length, min_question_length, max_answer_length, min_answer_length):
    res = []
    valid_sample = 0
    total_sample = 0
    with open(FLAGS.evaluate_file, 'r',encoding='utf-8') as f:
        json_obj = json.load(f)
    line_count = 0
    for qa in json_obj:
        question = qa['question']
        question_seg = cut_cut(question)# 问题 词序列

        if line_count > end_pos:
            break

        for ans in qa['passages']:
            total_sample = total_sample + 1
            # 样本数量限制
            answer = ans['content']
            line_count = line_count + 1
            if line_count > end_pos:
                break

            answer_seg = cut_cut(answer) # 问题 词序列
            # 答案长度过滤
            if len(question_seg) > max_question_length or len(question_seg) < min_question_length or \
            len(answer_seg) > max_answer_length or len(answer_seg) < min_answer_length:
                continue
            if ans['label'] == 0:
                label = [1, 0] 
            else:
                label = [0, 1]
            valid_sample = valid_sample + 1
            res.append((label, [ans['passage_id']], question_seg, answer_seg))
    return (total_sample, valid_sample, np.array(res))
 
"""
初始化sample && 词向量。json => list
"""
def init_sp(end_pos, max_question_length, min_question_length, max_answer_length, min_answer_length):
    print(max_question_length)
    print(min_question_length)
    print(max_answer_length)
    print(min_answer_length)
    res = []
    valid_sample = 0
    total_sample = 0
    with open(FLAGS.evaluate_file, 'r',encoding='utf-8') as f:
        json_obj = json.load(f)
    line_count = 0
    for qa in json_obj:
        question = qa['question']
        question_seg = cut_cut(question)# 问题 词序列

        if line_count > end_pos:
            break

        for ans in qa['passages']:
            total_sample = total_sample + 1
            # 样本数量限制
            answer = ans['content']
            line_count = line_count + 1
            if line_count > end_pos:
                break

            answer_seg = cut_cut(answer) # 问题 词序列
            # 答案长度过滤
            if len(question_seg) > max_question_length or len(question_seg) < min_question_length or \
              len(answer_seg) > max_answer_length or len(answer_seg) < min_answer_length:
                if ans['label'] == 0:
                    label = [1, 0] 
                else:
                    label = [0, 1]
                valid_sample = valid_sample + 1
                res.append((label, [ans['passage_id']], question_seg, answer_seg))
    return (total_sample, valid_sample, np.array(res))
 

"""
生成训练数据。分批生成，节约Menory。假定样本书不超过1 * 10^^8
"""
def batch_iter(data, batch_size, epoch_num, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(epoch_num):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            res = []
            for label, id, question, answer in shuffled_data[start_index:end_index]:
                res.append((label, id, convert_to_word_vector(question, FLAGS.max_question_length),
                            convert_to_word_vector(answer, FLAGS.max_answer_length)))
            yield np.array(res)

def batch_iter_sp(data, shuffle=True):
    print(len(data))
    data = np.array(data)
    data_size = len(data)
    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for label, id, question, answer in shuffled_data:
        ql = math.ceil(len(question[0:FLAGS.max_question_length]) / 64)
        al = math.ceil(len(answer) / 64)
        rows = max(ql, al)
        res = [] 
        for i in range(0, rows):
            res.append((label, id, convert_to_word_vector(question[0:FLAGS.max_question_length], FLAGS.max_question_length),
                convert_to_word_vector(answer[FLAGS.max_answer_length * i: FLAGS.max_answer_length * (i + 1)], FLAGS.max_answer_length)))
        yield np.array(res)

manual_input = False
if manual_input == False:
    total_sample, valid_sample, text_data = init(1000000, max_question_length=FLAGS.max_question_length,
                                                        min_question_length=0,
                                                        max_answer_length=FLAGS.max_answer_length,
                                                        min_answer_length=0
                                                        )


    _, _, data_sp = init_sp(1000000, max_question_length=FLAGS.max_question_length, 
                                                        min_question_length=0,
                                                        max_answer_length=FLAGS.max_answer_length,
                                                        min_answer_length=0,
                                                        )

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        # input_x = graph.get_operation_by_name("input_x").outputs[0]
        input_questions = graph.get_operation_by_name("questions").outputs[0]
        input_answers = graph.get_operation_by_name("answers").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Collect the predictions here
        ids = []
        scores = []
        all_labels = []
        if manual_input == True:
            while True:
                question = input("Question")
                answer = input("Answer")
                question_seg = cut_cut(question)
                answer_seg = cut_cut(answer)
                question_vec = convert_to_word_vector(question_seg, 50)
                answer_vec = convert_to_word_vector(answer_seg, 64)
                print(sess.run(predictions, {input_questions: [question_vec], input_answers: [answer_vec]}))
        else:
            # Generate batches for one epoch
            batches = batch_iter(text_data, FLAGS.batch_size, 1, shuffle=True)
            for i, batch in enumerate(batches):
                labels, id, questions, answers = zip(*batch)
                batch_predictions = sess.run(predictions, {input_questions: questions, input_answers: answers})
                all_predictions = np.concatenate([all_predictions, batch_predictions])
                all_labels = all_labels + np.array(np.argmax(labels, 1)).tolist()
                ids = ids + np.array(id).tolist()
                acc = np.average(np.equal(batch_predictions, np.argmax(labels, 1)).astype(np.float32))
                print("[{}] step {} batch_size: {}  accuracy: {}".format(_now(), i, len(labels), acc))
                scores.append(acc)
            print("[{}] start evaluate special data ...".format(_now()))
            batches = batch_iter_sp(data_sp, shuffle=True)
            for i, batch in enumerate(batches):
                labels, id, questions, answers = zip(*batch)
                batch_predictions = sess.run(predictions, {input_questions: questions, input_answers: answers})
                batch_predictions = [1, ] if np.sum(batch_predictions) > 0 else [0, ]
                all_predictions = np.concatenate([all_predictions, batch_predictions])
                all_labels = all_labels + np.array(np.argmax(labels, 1)[0:1]).tolist()
                ids = ids + np.array(id[0:1]).tolist()
                acc = np.average(np.equal(batch_predictions, np.argmax(labels[0:1], 1)).astype(np.float32))
                print("[{}] step {} batch_size: {}  accuracy: {}".format(_now(), i, len(labels), acc))
                scores.append(acc)

print("Accuracy: {}".format(np.average(scores)))

# Save the evaluation to a csv
print(len(all_labels))
print(len(all_predictions))

predictions_human_readable = np.column_stack((ids, all_predictions, all_labels))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
