import copy
import sys
import random
import csv
import numpy as np
import os

class EM:

    def __init__(self, e2wl, w2el, label_set):

        self.e2wl = e2wl
        self.w2el = w2el
        self.label_set = label_set

        ###################################################################
        # Expectation Maximization
        ###################################################################

    def InitPj(self):
        l2pd = {}
        for label in self.label_set:
            l2pd[label] = 1.0 / len(self.label_set)
        return l2pd

    def InitWM(self, workers):
        wm = {}

        if workers == {}:
            workers = self.w2el.keys()
            for worker in workers:
                wm[worker] = 0.8
        else:
            for worker in workers:
                if worker not in wm:  # workers --> wm
                    wm[worker] = 0.8
                else:
                    wm[worker] = workers[worker]

        return wm

    # E-step
    def ComputeTij(self, e2wl, l2pd, wm):
        e2lpd = {}
        for e, workerlabels in e2wl.items():
            e2lpd[e] = {}
            for label in self.label_set:
                e2lpd[e][label] = 1.0  # l2pd[label]

            for worker, label in workerlabels:
                for candlabel in self.label_set:
                    if label == candlabel:
                        e2lpd[e][candlabel] *= wm[worker]
                    else:
                        e2lpd[e][candlabel] *= (1 - wm[worker]) * 1.0 / (len(self.label_set) - 1)

            sums = 0
            for label in self.label_set:
                sums += e2lpd[e][label]

            if sums == 0:
                for label in self.label_set:
                    e2lpd[e][label] = 1.0 / self.len(self.label_set)
            else:
                for label in self.label_set:
                    e2lpd[e][label] = e2lpd[e][label] * 1.0 / sums

        # print e2lpd
        return e2lpd

    # M-step
    def ComputePj(self, e2lpd):
        l2pd = {}

        for label in self.label_set:
            l2pd[label] = 0
        for e in e2lpd:
            for label in e2lpd[e]:
                l2pd[label] += e2lpd[e][label]

        for label in self.label_set:
            l2pd[label] = l2pd[label] * 1.0 / len(e2lpd)

        return l2pd

    def ComputeWM(self, w2el, e2lpd):
        wm = {}
        for worker, examplelabels in w2el.items():
            wm[worker] = 0.0
            for e, label in examplelabels:
                wm[worker] += e2lpd[e][label] * 1.0 / len(examplelabels)

        return wm

    def Run(self, iter, workers={}):
        # wm     worker_to_confusion_matrix = {}
        # e2pd   example_to_softlabel = {}
        # l2pd   label_to_priority_probability = {}

        l2pd = self.InitPj()
        wm = self.InitWM(workers)
        while iter > 0:
            # E-step
            e2lpd = self.ComputeTij(self.e2wl, {}, wm)

            # M-step
            # l2pd = self.ComputePj(e2lpd)
            wm = self.ComputeWM(self.w2el, e2lpd)

            # print l2pd,wm

            iter -= 1

        return e2lpd, wm


def getaccuracy(truthfile, e2lpd, label_set):
    e2truth = {}
    f = np.loadtxt(open(truthfile))
    reader = f.astype(int)

    for line in reader:
        example, truth = line
        e2truth[example] = truth

    tcount = 0
    count = 0

    for e in e2lpd:

        if e not in e2truth:
            continue

        temp = 0
        for label in e2lpd[e]:
            if temp < e2lpd[e][label]:
                temp = e2lpd[e][label]

        candidate = []

        for label in e2lpd[e]:
            if temp == e2lpd[e][label]:
                candidate.append(label)

        truth = random.choice(candidate)

        count += 1

        if truth == e2truth[e]:
            tcount += 1

    return tcount * 1.0 / count


def gete2wlandw2el(datafile):
    e2wl = {}
    w2el = {}
    label_set = []

    f = np.loadtxt(open(datafile))
    reader = f.astype(int)

    for line in reader:

        worker, example, label = line
        if example not in e2wl:
            e2wl[example] = []
        e2wl[example].append([worker, label])

        if worker not in w2el:
            w2el[worker] = []
        w2el[worker].append([example, label])

        if label not in label_set:
            label_set.append(label)

    return e2wl, w2el, label_set


def gete2t(known_true):
    e2truth = {}
    f = np.loadtxt(open(known_true))
    reader = f.astype(int)

    for line in reader:
        example, truth = line
        e2truth[example] = truth

    return e2truth


if __name__ == '__main__':

    datafile = sys.argv[1]
    truthfile = sys.argv[2]


    e2wl, w2el, label_set = gete2wlandw2el(datafile)
    e2t = gete2t(truthfile)
    e2lpd, wm = EM(e2wl, w2el, label_set).Run(20)
    # print(wm)
    # print(e2lpd)

    accuracy = getaccuracy(truthfile, e2lpd, label_set)
    print(accuracy)

    inf_label = []
    for i in sorted(e2lpd):
        for j, p in e2lpd[i].items():
            if p == max(e2lpd[i].values()):
                # print(int(i), int(j))
                inf_label.append(int(j))

    # 如果文件存在，则先删除
    filename = './ZenCrowd.txt'
    if os.path.exists(filename):
        os.remove(filename)

    inference_label = open("./ZenCrowd.txt", "a+")
    for i in range(len(inf_label)):
        inference_label.write(str(i + 1) + ' ' + str(inf_label[i]) + '\n')
