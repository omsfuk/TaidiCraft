# -*- encoding: utf-8 -*-
import datetime
import json
import numpy as np

def _now():
    return datetime.datetime.now().strftime('%b-%d-%y %H:%M:%S')

# 维度扩充(fill with zero)
def expand_array(arr, dest_length):
    assert len(arr) <= dest_length
    ans = np.zeros((dest_length))
    ans[0:len(arr)] = np.array(arr)
    return ans

