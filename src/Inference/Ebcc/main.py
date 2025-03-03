import random
from data_pipeline import gete2wlandw2el,getaccuracy
import time
from results_summary import OfflineResultsSummary
from ebcc import ebcc_vb

random.seed(1)


if __name__ == '__main__':

    datasets = [
        ('situation1', 'situation1'),
        # ('crowdscale2013/sentiment', 'senti'),
        # ('crowdscale2013/fact_eval', 'fact'),
        # ('active-crowd-toolkit/CF', 'CF'),
        # ('active-crowd-toolkit/CF_amt', 'CF_amt'),
        # ('active-crowd-toolkit/MS', 'MS'),
        # ('crowd_truth_inference/s4_Face Sentiment Identification', 'face'),
        # ('crowd_truth_inference/s5_AdultContent', 'adult'),
        # ('SpectralMethodsMeetEM/dog', 'dog'),
        # ('SpectralMethodsMeetEM/web', 'web'),
        # ('mill','mill'),
        #
        # ('active-crowd-toolkit/SP', 'SP'),
        # ('active-crowd-toolkit/SP_amt', 'SP_amt'),
        # ('active-crowd-toolkit/ZenCrowd_all', 'ZC_all'),
        # ('active-crowd-toolkit/ZenCrowd_in', 'ZC_in'),
        # ('active-crowd-toolkit/ZenCrowd_us', 'ZC_us'),
        # ('crowd_truth_inference/d_jn-product', 'product'),
        # ('crowd_truth_inference/d_sentiment', 'tweet'),
        # ('SpectralMethodsMeetEM/bluebird', 'bird'),
        # ('SpectralMethodsMeetEM/rte', 'rte'),
        # ('SpectralMethodsMeetEM/trec', 'trec'),
    ]



    results = OfflineResultsSummary('EBCC')
    num_round = 40
    seed = 1
    for dataset, abbrev in datasets:
        print(dataset, abbrev)
        label_path = "../data/"+dataset+"/label.csv"
        truth_path = "../data/"+dataset+"/truth.csv"
        random.seed(seed)
        list_ = []

        e2wl, w2el, label_set = gete2wlandw2el(label_path)

        for round in range(num_round):
            print('dataset',abbrev,'round',round+1)
            t1 = time.time()
            truths,elbo, iterations = ebcc_vb(e2wl,w2el,label_set)
            t2 = time.time()
            acc = getaccuracy(truth_path, truths)
            results.add(abbrev,acc,t2-t1,iterations,elbo)

    print('EBCC results')
    print(results.get_EBCC_dataframe())
