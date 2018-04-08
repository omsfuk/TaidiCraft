# -*- encoding:utf-8 -*-
import json
import jieba
import re
import os
import sys

input_file = sys.argv[1]
question_limit = int(sys.argv[2])
pos_limit = int(sys.argv[3])
neg_limit = int(sys.argv[4])
output_file = "{}_{}_{}_{}.json".format(os.path.splitext(input_file)[0], question_limit, pos_limit, neg_limit)

ratio = pos_limit / neg_limit
question_length = 128
answer_length = 64
p = re.compile('[\u4e00-\u9fa50-9a-zA-Z]+')

print("question count: {}".format(question_limit))
print("pos_limit:      {}".format(pos_limit))
print("neg_limit:      {}".format(neg_limit))

jieba.initialize()

with open(input_file, "r") as f:
    q_list = json.load(f)

def gcd(x, y):
    for i in range(min(x, y), 0, -1):
        if ((x % i == 0) and (y % i == 0)):
            return i

q_count = 0
sample_count = 0
ans = []
gcd_pos = pos_limit // gcd(pos_limit, neg_limit)
gcd_neg = neg_limit // gcd(pos_limit, neg_limit)
for q in q_list:
    pos_count = 0
    neg_count = 0
    pos = []
    neg = []
    if len([1 for _ in filter(lambda x: p.match(x), jieba.lcut(q['question'], cut_all=False))]) > question_length:
        continue
    if q_count == question_limit:
        break
    q_count += 1
    for a in q['passages']:
        if len([1 for _ in filter(lambda x: p.match(x), jieba.lcut(a['content'], cut_all=False))]) > answer_length:
            continue
        if a['label'] == 1:
            pos.append(a)
        else:
            neg.append(a)

    if pos_limit <= len(pos) and neg_limit <= len(neg):
        pos = pos[0:pos_limit]
        neg = neg[0:neg_limit]
    else:
        pos = pos[0:min(len(pos) // gcd_pos, len(neg) // gcd_neg) * gcd_pos]
        neg = neg[0:min(len(pos) // gcd_pos, len(neg) // gcd_neg) * gcd_neg]
    if (len(pos) == 0 and pos_limit != 0) or (len(neg) == 0 and neg_limit != 0):
        continue
    sample_count += len(neg) + len(pos)
    ans.append({"question": q['question'], 'passages': neg + pos})

print("total question {}".format(q_count))
print("total sample   {}".format(sample_count))
print("Persistance to {}.....".format(output_file))
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(ans, f, ensure_ascii=False, indent=4)
print("Done")

