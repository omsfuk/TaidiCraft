# -*- encoding:utf-8 -*-
import json
import jieba
import sys
import os

input_file = sys.argv[1]
question_limit = int(sys.argv[2])
output_file = "unbalance_{}_{}.json".format(os.path.splitext(input_file)[0], question_limit)

pos_percentage = 0.5

with open(input_file, "r", encoding='utf-8') as f:
    obj = json.load(f)

res = []
count = 0
for qa in obj:
    if count >= question_limit:
        break
    answers = []
    for ans in qa['passages']:
        if count > question_limit:
            break
        count += 1
        if count > question_limit:
            break
        answers.append(ans)

    res.append({'question':qa['question'], 'passages': answers})
with open(output_file, "w", encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=4)


