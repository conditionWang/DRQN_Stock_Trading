import datetime
from collections import OrderedDict, deque
import math
import numpy as np
import json, codecs

def parse_time(text):    
    year = int(text[0:4])
    month = int(text[5:7])
    day = int(text[8:10])
    
    hour = int(text[11:13])
    mins = int(text[14:16])
    sec = 0
    return datetime.datetime(year, month, day, hour, mins, sec)

# change to time with interval of 15 mins
def map_datetime(dt):
    dt0 = dt
    return dt0.replace(minute=((dt.minute // 30)) * 30, second=0)

def get_ohlc(bucket):
    o, c = bucket[0], bucket[-1]
    h = max(bucket, key=lambda a: (a[1] + a[2]) / 2.0)
    l = min(bucket, key=lambda a: (a[1] + a[2]) / 2.0)
    return o, h, l, c

def calc_z_scores_parameters(cluster):
    cluster0 = np.asarray(cluster)
    mean = np.mean(cluster0, axis=0)
    variance = np.var(cluster0, axis=0)
    return mean, variance

def z_transform(value, mean, variance):
    result = (np.asarray(value) - mean) / variance
    return result.tolist()

def save_data_structure(structure, file):
    json.dump(structure, codecs.open(file, 'w', encoding='utf-8'), sort_keys=True, indent=4)