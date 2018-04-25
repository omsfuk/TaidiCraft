# -*- encoding:utf-8 -*-
import numpy as np
import tensorflow as tf
import os
import pickle
import sys
import csv
from lib_craft import mprint

os.environ["CUDA_VISIBLE_DEVICES"] = ""
# test_file = "data/testing_1000_4_1.vec"
test_file = sys.argv[1]
batch_size = 128 
checkpoint_dir = "runs/1523606604/checkpoints"

def get_sample(test_file):
    mprint("Load vector ...")
    with open(test_file, "rb") as f:
        data_dev = pickle.load(f)
    mprint("Complete")
    return data_dev

"""
生成数据
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
            yield shuffled_data[start_index:end_index]

data_dev = get_sample(test_file)

print("\nEvaluating...\n")

checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto()
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_questions = graph.get_operation_by_name("questions").outputs[0]
        input_questions_feature = graph.get_operation_by_name("question_feature").outputs[0]
        input_answers = graph.get_operation_by_name("answers").outputs[0]
        input_answers_feature = graph.get_operation_by_name("answer_feature").outputs[0]
        input_dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("accuracy/predictions").outputs[0]

        # Collect the predictions here
        ids = []
        scores = []
        all_labels = []
        all_predictions = []
       
        # Generate batches for one epoch
        batches = batch_iter(data_dev, batch_size, 1, shuffle=True)
        for i, batch in enumerate(batches):
            passage_id, labels, questions, answers, question_feature, answer_feature = zip(*batch)
            batch_predictions = sess.run(predictions, 
                    {input_questions: questions, input_answers: answers, input_dropout_keep_prob:1.0, 
                        input_questions_feature: question_feature, input_answers_feature: answer_feature})
            # all_predictions = np.concatenate((all_predictions, batch_predictions))
            all_predictions = all_predictions + np.array(batch_predictions).reshape((-1)).tolist()
            all_labels = all_labels + np.array(np.argmax(labels, 1)).tolist()
            ids = ids + np.array(passage_id).tolist()
            acc = np.average(np.equal(batch_predictions, np.argmax(labels, 1)).astype(np.float32))
            mprint("step {} batch_size: {}  accuracy: {}".format(i, len(labels), acc))
            scores.append(acc)

print("Accuracy: {}".format(np.average(scores)))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((ids, all_predictions, all_labels))
out_path = os.path.join(checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
