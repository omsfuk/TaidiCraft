# -*- encoding: utf-8 -*-
import os
import re
import sys
import json
import jieba
import gensim
import pickle
import numpy as np
import jieba.analyse
from lib_craft import mprint

input_file = sys.argv[1]

debug_mode = True # enbale/disable debug mode
print_every = 5000 # in debug mode, print debug info erery time 'print_every' q/a are finished
feature_vector_length = 10
question_vector_length = 50
answer_vector_length = 32 

def init():
    global p, filter_punt, dic, model
    p = re.compile(r'[\u4e00-\u9fa5_a-zA-Z0-9]+')
    filter_punt = lambda s: p.match(s)

    # 索引词典
    if os.path.isfile("word.index"):
        with open("word.index", "rb") as f:
            dic = pickle.load(f)
    else:
        raise SystemExit("Can't find word.index")

    mprint("Loading model ...")
    model = gensim.models.Word2Vec.load('npy/word2vec_wx')
    mprint("Model loading finished")

def extract_feature_vector(raw_sample, length):
    return ()

def get_segments(text, use_jieba_fenci, length):
    if use_jieba_fenci == True:
        return jieba.lcut(text, cut_all=False)
    else:
        return [x for x in filter(filter_punt, [w for w in jieba.analyse.extract_tags(text, topK=length)])]

def convert2vec(segments, d_len):
    result = np.zeros((d_len))
    for i, segment in enumerate(segments):
        if segment in dic.keys():
            result[i] = dic[segment]
    return result

def is_data_invalid(raw_sample):
    return raw_sample

def extract_sample(filename):
    with open(filename, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)
    for qa_pair in raw_samples:
        question, answers = qa_pair['question'], qa_pair['passages']
        for raw_answer in answers:
            passage_id, answer = raw_answer['passage_id'], raw_answer['content']
            # if the label does't exist, the defaut value is 0
            label = raw_answer.get('label', 0)
            yield (passage_id, label, question, answer)

def polish_sample(raw_sample):
    return raw_sample

def persistent(obj):
    output_file = "{}.vec".format(os.path.splitext(input_file)[0])
    mprint("Writing to <{}>. Please don't interrupt it ...".format(output_file))
    with open(output_file, "wb") as f:
        pickle.dump(obj, f)
    mprint("Done !")

init()
result = []
for count, raw_sample in enumerate(extract_sample(input_file)):
    if is_data_invalid(raw_sample) == False:
        continue
    if count % print_every == 0:
        mprint("Processing {} question/answer".format(count))
    raw_sample = polish_sample(raw_sample)
    passage_id, label, question, answer = raw_sample
    feature_vector = extract_feature_vector(raw_sample, feature_vector_length)
    label_vector = [0, 1] if label == 1 else [1, 0]
    question_segments = get_segments(question, use_jieba_fenci=True, length=question_vector_length)
    answer_segments = get_segments(answer, use_jieba_fenci=False, length=answer_vector_length)
    question_vector = convert2vec(question_segments, question_vector_length)
    answer_vector = convert2vec(answer_segments, answer_vector_length)
    result.append((passage_id, label_vector, question_vector, answer_vector) + feature_vector)

persistent(result)




