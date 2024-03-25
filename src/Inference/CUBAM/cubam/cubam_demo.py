import os, pickle,csv
from numpy import random, mean, std, sqrt
from Binary1dSignalModel import Binary1dSignalModel
import sys
import numpy as np
class cubam:
    def __init__(self,e2wl,w2el,label_set,nAnnotation):
        self.e2wl = e2wl
        self.w2el = w2el
        self.nAnnotation = nAnnotation
        self.workers = dict()
        self.examples = dict()
        self.labels = dict()
        i = 0
        for worker in self.w2el.keys():
            self.workers[worker]=i
            i = i + 1
        i = 0
        for example in self.e2wl.keys():
            self.examples[example]=i
            i = i + 1

        i = 0
        for label in label_set:
            self.labels[label]=i
            i = i + 1

    def Run(self):

        e2lpd=dict()

        fout = open("temp.txt", 'w')
        fout.write("%d %d %d\n" % (len(self.examples), len(self.workers),self.nAnnotation))
        for example, worker_label_set in self.e2wl.items():
            for (worker, label) in worker_label_set:
                fout.write("%d %d %d\n" % (int(self.examples[example]), int(self.workers[worker]),int(self.labels[label])))
        fout.close()


        getParameter = lambda prmdict, pidx: [prmdict[i][pidx] for i \
                                          in range(len(prmdict))]
        # m = BinaryNdSignalModel(filename="temp.txt", dim=2)
        # if using BinaryNdSignalModel, dimention of exi will be dim*n_example
        m = Binary1dSignalModel(filename="temp.txt")
        m.optimize_param()
        exi = getParameter(m.get_image_param(), 0)
        for (example, example_id) in self.examples.items():
            # only for 2-classificaiton
            if exi[example_id] > 0 :
                predict_label_id = 1
            else:
                predict_label_id = 0

            lpd={}
            for (label, label_id ) in self.labels.items():
                if label_id == predict_label_id:
                    lpd[label] = 1
                else:
                    lpd[label] = 0

            e2lpd[example] = lpd


        return e2lpd
###################################
# The above is the EM method (a class)
# The following are several external functions
###################################

def getaccuracy(truthfile, e2lpd):
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

    return tcount*1.0/count


def gete2wlandw2el(datafile):
    e2wl = {}
    w2el = {}
    label_set=[]
    nAnnotation = 0
    f = np.loadtxt(open(datafile))
    reader= f.astype(int)

    for line in reader:
        nAnnotation = nAnnotation +1
        worker, example, label = line
        if example not in e2wl:
            e2wl[example] = []
        e2wl[example].append([worker,label])

        if worker not in w2el:
            w2el[worker] = []
        w2el[worker].append([example,label])

        if label not in label_set:
            label_set.append(label)

    return e2wl,w2el,label_set,nAnnotation

if __name__ == "__main__":

    datafile = sys.argv[1]
    e2wl,w2el,label_set, nAnnotation = gete2wlandw2el(datafile)
    e2lpd = cubam(e2wl,w2el,label_set,nAnnotation).Run()

    print(e2lpd)
    truthfile = sys.argv[2]
    accuracy = getaccuracy(truthfile, e2lpd)
    print(accuracy)

    inf_label = []
    for i in sorted(e2lpd):
        for j, p in e2lpd[i].items():
            if p == max(e2lpd[i].values()):
                # print(int(i), int(j))
                inf_label.append(int(j))

    # 如果文件存在，则先删除
    filename = './CUBAM.txt'
    if os.path.exists(filename):
        os.remove(filename)

    inference_label = open(filename, "a+")
    for i in range(len(inf_label)):
        inference_label.write(str(i + 1) + ' ' + str(inf_label[i]) + '\n')
