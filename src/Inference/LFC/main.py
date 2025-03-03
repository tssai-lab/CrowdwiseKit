import math
import random
import time

from results_summary import OfflineResultsSummary
from data_pipeline import gete2wlandw2el, getaccuracy
from copy import deepcopy
from l_LFCmulti import EM

def from_z_to_truths_multi(z_i, label_set):
    truths = {}
    for item in z_i.keys():
        votes = z_i.get(item)
        candidate = []
        max_ = -1
        for class_ in label_set:
            if votes.get(class_) is None:
                continue
            if votes.get(class_) > max_:
                candidate.clear()
                candidate.append(class_)
                max_ = votes.get(class_)
            elif votes.get(class_) == max_:
                candidate.append(class_)
        truths[item] = random.choice(candidate)
    return truths

if __name__ == '__main__':
    if __name__ == "__main__":

        datasets = [
            ('crowdscale2013/sentiment', 'senti'),
            ('crowdscale2013/fact_eval', 'fact'),
            ('active-crowd-toolkit/CF', 'CF'),
            ('active-crowd-toolkit/CF_amt', 'CF_amt'),
            ('active-crowd-toolkit/MS', 'MS'),
            ('crowd_truth_inference/s4_Face Sentiment Identification', 'face'),
            ('crowd_truth_inference/s5_AdultContent', 'adult'),
            ('SpectralMethodsMeetEM/dog', 'dog'),
            ('SpectralMethodsMeetEM/web', 'web'),
            ('mill', 'mill'),

            ('active-crowd-toolkit/SP', 'SP'),
            ('active-crowd-toolkit/SP_amt', 'SP_amt'),
            ('active-crowd-toolkit/ZenCrowd_all', 'ZC_all'),
            ('active-crowd-toolkit/ZenCrowd_in', 'ZC_in'),
            ('active-crowd-toolkit/ZenCrowd_us', 'ZC_us'),
            ('crowd_truth_inference/d_jn-product', 'product'),
            ('crowd_truth_inference/d_sentiment', 'senti_1k'),
            ('SpectralMethodsMeetEM/bluebird', 'bird'),
            ('SpectralMethodsMeetEM/rte', 'rte'),
            ('SpectralMethodsMeetEM/trec', 'trec'),

        ]

        results = OfflineResultsSummary('LFC')
        num_round = 20
        seed = 1
        random.seed(seed)
        for dataset, abbrev in datasets:

            label_path = "data/" + dataset + "/label.csv"
            truth_path = "data/" + dataset + "/truth.csv"
            random.seed(seed)
            list_ = []

            e2wl, w2el, label_set = gete2wlandw2el(label_path)

            for round in range(num_round):
                print(abbrev, 'round', round + 1)
                t1 = time.time()
                e2lpd, w2cm = EM(e2wl, w2el, label_set).Run()
                truths = from_z_to_truths_multi(e2lpd,label_set)
                t2 = time.time()
                acc = getaccuracy(truth_path, truths)
                results.add(abbrev, acc, t2 - t1, 20)

        print('LFC results')
        print(results.to_dataframe_mean())

    for ele in results.to_dataframe_mean()[['dataset','runtime mean']]:
        print(ele)

