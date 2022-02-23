import json
import math
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing, metrics as sk_metrics
from tqdm import tqdm

# The test data directory
# ProgramData
#  |-1
#  | |- 1.c
#  | |- 2.c
#  | |- ...
#  |-2
# ...
test_dir = r"../test/ProgramData/"
use_progress_bar = True

total_count = 0


def load_pairwise_compare_data(p, scale=True):
    global test_dir, total_count, use_progress_bar
    # buffer the data
    buffer_file = "buffer/%s.json" % p.get_parser_name()
    try:
        f = open(buffer_file, "r")
        metrics_list = json.loads(f.read())
        f.close()
        print("Read Buffer %s" % buffer_file)
        if use_progress_bar:
            total_count = 0
            for row in metrics_list:
                total_count += len(row)
        return metrics_list
    except:
        print("Buffer File Not Found Or Invalid, Generating...")

    g = os.walk(test_dir)
    sample_set = []
    for path_dir, dir_list, _ in g:
        for dir_name in dir_list:
            g2 = os.walk(os.path.join(path_dir, dir_name))
            for path, _, file_list in g2:
                filenames = []
                for file_name in file_list:
                    filenames.append(os.path.join(path, file_name))
                sample_set.append(filenames)

    metrics_belongs = []
    metrics = []
    i = 0
    for filenames in sample_set:
        i += 1
        for filename in filenames:
            f = open(filename)
            try:
                c_text = f.read()
            except:
                print("Cannot Open File %s" % filename)
                continue
            v = p.get_parser()
            v.parse(c_text)
            metrics_belongs.append(i - 1)
            metrics.append(v.get_array().tolist())

    if use_progress_bar:
        total_count = len(metrics)

    if scale:
        metrics = preprocessing.scale(np.array(metrics))

    metrics_list = []
    for j in range(0, np.max(metrics_belongs) + 1):
        metrics_list.append([])
    for j in range(0, len(metrics)):
        # the metrics[j]
        if scale:
            # is a numpy array
            metrics_list[metrics_belongs[j]].append(metrics[j].tolist())
        else:
            # is a list
            metrics_list[metrics_belongs[j]].append(metrics[j])

    f = open(buffer_file, "w")
    f.write(json.dumps(metrics_list))
    f.close()

    return metrics_list  # list[list]


def pairwise_compare_data(p, metrics_list):
    same = []
    not_same = []

    global total_count, use_progress_bar
    pbar = None
    if use_progress_bar:
        pbar = tqdm(total=(total_count + 1) * total_count / 2, unit="cmp", leave=True)

    for i in range(0, len(metrics_list)):
        for j in range(0, len(metrics_list[i])):
            for k in range(j, len(metrics_list[i])):
                same.append(p.similarity(np.array(metrics_list[i][j]), np.array(metrics_list[i][k])))

                if use_progress_bar:
                    pbar.update(1)

            for k in range(i + 1, len(metrics_list)):
                for l in range(0, len(metrics_list[k])):
                    not_same.append(p.similarity(np.array(metrics_list[i][j]), np.array(metrics_list[k][l])))

                    if use_progress_bar:
                        pbar.update(1)

    return same, not_same


def similarity_roc(p, metrics_list, show_hist=False, show_roc=False):
    same, not_same = pairwise_compare_data(p, metrics_list)

    if show_hist:
        plt.hist(np.array(same))
        plt.show()
        plt.hist(np.array(not_same))
        plt.show()

    y = [1] * len(same) + [0] * len(not_same)
    score = []
    score.extend(same)
    score.extend(not_same)
    fpr, tpr, thresholds = sk_metrics.roc_curve(np.array(y), np.array(score))
    auc_score = sk_metrics.auc(fpr, tpr)

    if show_roc:
        plt.figure()
        plt.plot(fpr, tpr)
        plt.show()

    threshold = 0.0
    threshold_dis = 0.0
    threshold_fpr = 0.0
    threshold_tpr = 0.0
    for i in range(0, len(fpr)):
        dis = abs((fpr[i] - tpr[i]) / math.sqrt(2))
        if dis > threshold_dis:
            threshold = thresholds[i]
            threshold_fpr = fpr[i]
            threshold_tpr = tpr[i]
            threshold_dis = dis

    return threshold, threshold_tpr, threshold_fpr, auc_score
