# -*- encoding: utf-8 -*-
import json
import jieba
import gensim
import csv
import codecs
import re
import sys
from gensim.models import word2vec

input_file = sys.argv[1]

dic = {}
model = gensim.models.Word2Vec.load('./npy/word2vec_wx')

with open(input_file, 'r') as f:
    obj = json.load(f)

positive_sample = 0
negative_sample = 0
missing_word = 0
existing_word = 0
longest_question = 0
longest_answer = 0

p = re.compile('[\u4e00-\u9fa50-9a-zA-Z]')

"""
axis True 问题 False 答案
"""
def process(sequence, axis):
    global missing_word
    global longest_question
    global longest_answer
    global existing_word
    index = 0
    seg_list = [x for x in filter(lambda x: p.match(x), jieba.lcut(sequence, cut_all=False))]
    if axis == True:
        if len(seg_list) > longest_question:
            longest_question = len(seg_list)
    else:
        if len(seg_list) > longest_answer:
            longest_answer = len(seg_list)

    for seg in seg_list:
        if seg not in dic.keys():
            dic[seg] = 1 
            index = index + 1
            if seg not in model.wv.vocab:
                existing_word = existing_word + 1
            else:
                missing_word = missing_word + 1
        else:
            dic[seg] += 1
total_sample = 0
for qa in obj:
    process(qa['question'], True)
    for ans in qa['passages']:
        total_sample += 1
        if ans['label'] == 1:
            process(ans['content'], False)
            positive_sample = positive_sample + 1
        else:
            negative_sample = negative_sample + 1

def dict2csv(dic):
    with codecs.open('word_frequency_question.csv', 'w', "utf_8_sig") as csvfile:
        csv.writer(csvfile).writerows(dic.items())

dict2csv(dic)

print("total sample    {}".format(total_sample))
print("positive sample {}".format(positive_sample))
print("negative sample {}".format(negative_sample))
print("total word      {}".format(len(dic)))
print("missing word    {}".format(missing_word))
print("existing word   {}".format(existing_word))
print("longest question{}".format(longest_question))
print("longest answer  {}".format(longest_answer))


