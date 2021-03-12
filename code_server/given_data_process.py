import datetime
from collections import OrderedDict, deque
import math
import numpy as np
import json, codecs
from data_process_common import *
import os

file_path = './historical_price/'
datafile_list = os.listdir(file_path)
# print(datafile_list)

# give the file name of data set
# datafile_list = ['../historical_price/AAL_2015-12-30_2021-02-21_minute.csv', '../historical_price/AAPL_2015-12-30_2021-02-21_minute.csv']
time_list = []
open_price_list = []
close_price_list = []

for i in datafile_list:
    time = []
    open_price = []
    close_price = []
    path_i = os.path.join(file_path, i)
    print(path_i)
    with open(path_i) as f:
        lines = f.readlines()
    for j in range(len(lines) - 1):
        line = lines[j + 1]
        line_split = line.split(",")
        open_price.append(float(line_split[3]))
        close_price.append(float(line_split[4]))
        # the time is 7th colomn
        time.append(parse_time(line_split[7]))

    time, open_price, close_price = np.array(time), np.array(open_price), np.array(close_price)

    time_list.append(time)
    open_price_list.append(open_price)
    close_price_list.append(close_price)

# print(time_list[0][0], open_price_list[0][0], close_price_list[0][0])

for index in range(len(datafile_list)):
    # define buckets of prices [(t1_str, [(t1, p1), (t2, p2), (t3, p3)]), (t2_str, [(t4, p4), (t5, p5), (t6, p6)]), ...]
    buckets = OrderedDict()
    for t, o_p, c_p in zip(time_list[index], open_price_list[index], close_price_list[index]):
        printed_time = str(map_datetime(t))
        if printed_time not in buckets:
            buckets[printed_time] = []
            
        buckets[printed_time].append((t, o_p, c_p))

    # print(list(buckets.items())[1])

    # calculate ohlc data
    ohlc = OrderedDict()
    for t, bucket in buckets.items():
        ohlc[t] = get_ohlc(bucket)

    # print(list(ohlc.items())[1])

    closing = list(map(lambda t_v: (t_v[0], t_v[1][3][2]), ohlc.items()))
    # print(closing[0:5])

    # calculate 8-delayed-log-returns of closing prices
    n = 8
    log_returns = []
    lag = deque()
    last_price = None
    for t, v in closing:
        if last_price is not None:
            lag.append(math.log(v / last_price))
            while len(lag) > n:
                lag.popleft()
                
            if len(lag) == n:
                log_returns.append((t, list(lag)))
        last_price = v

    # print(log_returns[0:5])

    # z-score normalization, group 96 states into a cluster
    z_score_clusters = OrderedDict()
    for n, t_vs in enumerate(log_returns):
        i = n // 96
        if i not in z_score_clusters:
            z_score_clusters[i] = []
        z_score_clusters[i].append(t_vs[1])

    z_score_transformed = []
    for n, t_vs in enumerate(log_returns):
        i = n // 96
        mean, variance = calc_z_scores_parameters(z_score_clusters[i])
        z_score_transformed.append([t_vs[0], z_transform(t_vs[1], mean, variance)])

    save_data_structure(z_score_transformed, "./data_states/" + datafile_list[index][0:3] + "-states.json")
    save_data_structure(closing, "./data_closing/" + datafile_list[index][0:3] + "-closing.json")
    print('save file {}'.format(index))

