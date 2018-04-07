# -*- encoding:utf-8 -*-
import json
import jieba

sample_count = 1000
pos_percentage = 0.5

with open("training.json", "r", encoding='utf-8') as f:
    obj = json.load(f)

pos_count = 0
neg_count = 0
res = []
count = 0
for qa in obj:
    if len(jieba.lcut(qa['question'], cut_all=False)) > 50:
        continue
    if count >= sample_count:
        break
    for ans in qa['passages']:
        if len(jieba.lcut(ans['content'], cut_all=False)) > 64:
            continue
        if count > sample_count:
            break
        if pos_count < sample_count * pos_percentage and ans['label'] == 1:
            count += 1
            pos_count += 1
            res.append({'question': qa['question'], "passages": [{'passage_id': count,'label': ans['label'], 'content': ans['content']}]}) 
        if neg_count < sample_count * (1 - pos_percentage) and ans['label'] == 0:
            count += 1
            neg_count += 1
            res.append({'question': qa['question'], "passages": [{'passage_id': count,'label': ans['label'], 'content': ans['content']}]}) 
        print("neg: {}".format(neg_count))
        print("pos: {}".format(pos_count))

with open("mini_sample.json", "w", encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False)


