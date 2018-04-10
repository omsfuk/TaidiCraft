# -*- encoding: utf-8 -*-
import numpy as np
import json
import pickle
import os
import sys
import re
import gensim
import jieba.analyse
from lib_craft import mprint
# from textrank4zh import TextRank4Keyword, TextRank4Sentence

input_file = sys.argv[1]

enable_debug_mode = False
enable_generate_passage_id = True# 是否生成passage_id
vec_file = "{}.vec".format(os.path.splitext(input_file)[0])
# tr4w = TextRank4Keyword()
p = re.compile(r'[\u4e00-\u9fa5_a-zA-Z0-9]+')
filter_punt = lambda s: p.match(s)

mprint("Loading model ...")
model = gensim.models.Word2Vec.load('npy/word2vec_wx')
mprint("Model loading finished")

question_keywords_limit = 50
answer_keywords_limit = 32

# 索引词典
if os.path.isfile("word.index"):
    with open("word.index", "rb") as f:
        dic = pickle.load(f)
else:
    raise SystemExit("Can't find word.index")

def get_seg(text, num):
    return [x for x in filter(filter_punt, [w for w in jieba.analyse.extract_tags(text, topK=num)])]

def convert2vec(segs, d_len):
    res = np.zeros((d_len))
    for i, seg in enumerate(segs):
        if seg in dic.keys():
            res[i] = dic[seg]
    return res
        
def step_one(filename):
    mprint("Starting step one")
    res = []
    count = 0
    with open(filename, "r", encoding='utf-8') as f:
        obj = json.load(f)
    for qa in obj:
        question = qa['question']
        # question_seg = get_seg(question, question_keywords_limit)
        # 换用结巴分词。关键词信息丢失太大
        question_seg = jieba.lcut(question, cut_all=False)[0: question_keywords_limit]
        for ans in qa['passages']:
            count += 1
            if count % 5000 == 0:
                mprint("Process {} question/answer".format(count))
            answer = ans['content']
            # 如果不含标签，则生成0
            label = 0 if 'label' not in ans.keys() else ans['label']
            answer_seg = get_seg(answer, answer_keywords_limit)
            res.append((ans['passage_id'], label, question_seg, answer_seg))
    mprint("Complete.")
    return res

def step_two(raw_data):
    res = []
    count = 0
    for passage_id, label, question, answer in raw_data:
        count += 1
        if count % 5000 == 0:
            mprint("Process {} question/answer".format(count))
        l_vec = [1, 0] if label == 0 else [0, 1]
        q_vec = convert2vec(question, question_keywords_limit)
        a_vec = convert2vec(answer, answer_keywords_limit)
        if enable_generate_passage_id == True:
            res.append(([passage_id, ], l_vec, q_vec, a_vec))
        else:
            res.append((l_vec, q_vec, a_vec))
    # shuffle
    res = np.array(res)
    shuffle_indices = np.random.permutation(np.arange(len(res)))
    res = res[shuffle_indices]
    return res
raw_data = step_one(input_file)
res = step_two(raw_data)

mprint("Writing to <{}>. Please don't interrupt it ...".format(vec_file))
with open(vec_file, "wb") as f:
    pickle.dump(res, f)
mprint("Done !")


