# -*- encoding: utf-8 -*-
import datetime
import numpy as np

def _now():
    return datetime.datetime.now().strftime('%b-%d-%y %H:%M:%S')

def balance_sample(data):
    data = np.array(data)
    positive_sample = [x for x in filter(lambda x: x[0] == [0, 1], data)]
    negative_sample = [x for x in filter(lambda x: x[0] == [1, 0], data)]
    
    print("[{}] original sample: positive {} negative {}".format(_now(), len(positive_sample), len(negative_sample)))
    positive_sample = positive_sample[0:min(len(positive_sample), len(negative_sample))]
    negative_sample = negative_sample[0:min(len(positive_sample), len(negative_sample))]
    print("[{}] after balance sample: positive {} negative {}".format(_now(), len(positive_sample), len(negative_sample)))
    
    return np.array(positive_sample + negative_sample)

# 维度扩充(fill with zero)
def expand_array(arr, dest_length):
    assert len(arr) <= dest_length
    ans = np.zeros((dest_length))
    ans[0:len(arr)] = np.array(arr)
    return ans
    
