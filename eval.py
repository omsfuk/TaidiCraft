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
from lib_craft import expand_array
from lib_craft import _now
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_integer("max_question_length", 50, "最大问题长度")
tf.flags.DEFINE_integer("min_question_length", 2, "最小问题长度")
tf.flags.DEFINE_integer("max_answer_length", 64, "最大答案长度")
tf.flags.DEFINE_integer("min_answer_length", 5, "最小答案长度")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "runs/1522913279/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS

with open("word.index", "rb") as f:
    dic = pickle.load(f)
pattern = re.compile(r'[\u4e00-\u9fa5_a-zA-Z0-9１２３４５６７８９０]')
punct = set(u'''#ㄍ <>/\\[]:!)］∫,.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒
        ﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠
        々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻
        ︽︿﹁﹃﹙﹛﹝（｛“‘-—_…... ''')
filter_punt = lambda s: u''.join(filter(lambda x: True if pattern.match(x) and x not in punct else False , s))


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
def init(end_pos=100000000):
    res = []
    valid_sample = 0
    total_sample = 0
    with open('test_data_sample.json', 'r',encoding='utf-8') as f:
        json_obj = json.load(f)
    line_count = 0
    for qa in json_obj:
        question = qa['question']
        question_seg = jieba.lcut(filter_punt(question), cut_all=False) # 问题 词序列
        # 问题长度过滤
        if len(question_seg) > FLAGS.max_question_length or len(question_seg) < FLAGS.min_question_length:
            continue

        if line_count > end_pos:
            break

        for ans in qa['passages']:
            total_sample = total_sample + 1
            # 样本数量限制
            answer = ans['content']
            line_count = line_count + 1
            if line_count > end_pos:
                break

            answer_seg = jieba.lcut(filter_punt(answer), cut_all=False) # 问题 词序列
            # 答案长度过滤
            if len(answer_seg) > FLAGS.max_answer_length or len(answer_seg) < FLAGS.min_question_length:
                continue
            """
            if ans['label'] == 0:
                label = [1, 0] 
            else:
                label = [0, 1]
            """
            valid_sample = valid_sample + 1
            res.append(([ans['passage_id']], question_seg, answer_seg))
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
            for id, question, answer in shuffled_data[start_index:end_index]:
                res.append((id, convert_to_word_vector(question, FLAGS.max_question_length),
                            convert_to_word_vector(answer, FLAGS.max_answer_length)))
            yield np.array(res)

# CHANGE THIS: Load data. Load your own data here
"""
if FLAGS.eval_train:
    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    y_test = np.argmax(y_test, axis=1)
else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]
"""

total_sample, valid_sample, text_data = init(1600)

# Map data into vocabulary
"""
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))
"""

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

        # Generate batches for one epoch
        batches = batch_iter(text_data, FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []
        ids = []
        scores = []

        for i, batch in enumerate(batches):
            id, questions, answers = zip(*batch)
            batch_predictions = sess.run(predictions, {input_questions: questions, input_answers: answers})
            all_predictions = np.concatenate([all_predictions, batch_predictions])
            ids = ids + np.array(id).tolist()
            print("[{}] {{i}} batch_size: {}  accuracy: {}".format(_now(), i, len(questions)))
            """
            acc = np.average(np.equal(batch_predictions, np.argmax(labels, 1)).astype(np.float32))
            print("batch_size: {}  accuracy: {}".format(len(labels), acc))
            scores.append(acc)
            """
"""
# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
"""
# print("Accuracy: {}".format(np.average(scores)))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(ids), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
