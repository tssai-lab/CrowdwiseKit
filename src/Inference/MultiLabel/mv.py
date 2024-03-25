# -*- coding: utf-8 -*-
# Majority Voting (MV
# @author: Jing Zhang (jzhang@njust.edu.cn))

import numpy
import random
from Inference.MultiLabel.core import samplable, data, utils
from Inference.MultiLabel import model

class MVWorker:

    def __init__(self, worker):
        self.worker = worker
        self.M = 0
        self.K = 0
        self.pi_list = [None]

    def initialize(self, M, K):
        self.M = M
        self.K = K
        for m in range(1, self.M + 1):
            pi = numpy.ndarray(shape=(self.K + 1, self.K + 1), dtype=samplable.RealV, order='C')
            self.random_initialize_pi(pi, 0.7, 0.9)
            self.pi_list.append(pi)

    def random_initialize_pi(self, pi, diagonal_low, diagonal_high):
        # we get K diagonal elements randomly in the range of low - high
        diagonal = [None]
        for i in range(1, self.K + 1):
            val = random.uniform(diagonal_low, diagonal_high)
            diagonal.append(val)
        for i in range(1, self.K + 1):
            remainder = 1.0 - diagonal[i]
            parts = utils.split_val_rand(remainder, self.K - 1)
            parts_index = 0
            for j in range(1, self.K + 1):
                if i == j:
                    pi[i][j] = samplable.RealV(diagonal[i])
                else:
                    pi[i][j] = samplable.RealV(parts[parts_index])
                    parts_index += 1

    def compute_pi(self, instances):
        for m in range (1, self.M+1):
            curr_pi = numpy.ndarray(shape=(self.K + 1, self.K + 1), dtype=int, order='C')
            curr_pi.fill(0)
            for inst in instances:
                d = self.worker.get_label_val_for_inst(inst.inst.id, m)
                if d != 0:
                    curr_pi[inst.inst.get_integrated_label(m).val][d] += 1
            for k in range(1, self.K + 1):
                s = 0
                for q in range(1, self.K + 1):
                    s += curr_pi[k][q]
                if s != 0:
                    for d in range(1, self.K + 1):
                        self.pi_list[m][k][d].append(float(curr_pi[k][d]) / float(s))
                else:
                    for d in range(1, self.K + 1):
                        self.pi_list[m][k][d].append(self.pi_list[m][k][d].getV())
        #self.print_pis()

    def print_pis(self):
        for m in range(1, self.M + 1):
            print('pi (' + str(m) + ') of worker ' + str(self.worker.id) + ':')
            for i in range(1, self.K + 1):
                for j in range(1, self.K + 1):
                    self.pi_list[m][i][j].print_obj()
                print('')

class MVInstance:

    def __init__(self, inst):
        self.inst = inst


class MVModel(model.Model):
    """
    majority voting model for both single and multi-class
    """

    def __init__(self):
        model.Model.__init__(self)
        self.workers = []
        self.instances = []
        self.I = 0
        self.J = 0
        self.K = 0
        self.M = 0  # M is the number of label ids
        self.theta_list = [None]

    def initialize(self, dataset):
        self.M = dataset.get_label_id_size()
        print('The number of label ids = ' + str(self.M))
        self.I = dataset.get_instance_size()
        self.J = dataset.get_worker_size()
        print('Total ' + str(self.I) + ' instances and ' + str(self.J) + ' workers')
        self.initialK(dataset)
        print('K = ' + str(self.K))
        for i in range(1, self.I + 1):
            inst = dataset.get_instance(i)
            mv_inst = MVInstance(inst)
            self.instances.append(mv_inst)
        for w in range(1, self.J + 1):
            worker = dataset.get_worker(w)
            mv_worker = MVWorker(worker)
            mv_worker.initialize(self.M, self.K)
            self.workers.append(mv_worker)
        for m in range(1, self.M + 1):
            thetas = numpy.ndarray(shape=(self.K + 1), dtype=samplable.RealV, order='C')
            for i in range(1, self.K + 1):
                thetas[i] = samplable.RealV(1.0 / self.K)
            self.theta_list.append(thetas)

    def initialK(self, dataset):
        maxK = 0
        for k in range(1, self.M + 1):
            currentK = dataset.get_label_val_size(k)
            if  currentK  > maxK:
                maxK = currentK
        self.K = maxK

    def infer(self, dataset, soft=False):
        self.initialize(dataset)
        for inst_id in range(1, self.I + 1):
            inst = dataset.get_instance(inst_id)
            label_ids = inst.get_label_id_list()
            for label_id in label_ids:
                labels = inst.get_noisy_labels(label_id)
                num_class = dataset.get_label_val_size(label_id)
                voted = self._vote(labels, num_class)
                # set integrated label
                integrated_label = data.Label(label_id)
                integrated_label.inst_id = inst.id
                integrated_label.worker_id = data.Worker.AGGR
                integrated_label.val = voted
                inst.add_integrated_label(integrated_label)
        self.parameter_compute()

    def _vote(self, labels, num_class):
        counts = dict()
        for k in range(1, num_class + 1):
            counts.setdefault(k, 0)
        for label in labels:
            counts[label.val] += 1
        # find max number
        maxval = counts[1]
        maxindex = 1
        for (k, v) in counts.items():
            if v > maxval:
                maxval = v
                maxindex = k
        # find multiple max value
        maxlist = list()
        maxlist.append(maxindex)
        for (k, v) in counts.items():
            if v == maxval:
                maxlist.append(k)
        pos = numpy.random.randint(0, len(maxlist))
        return maxlist[pos]

    def parameter_compute(self):
        for m in range(1, self.M + 1):
            numlist = []
            for k in range(0, self.K + 1):
                numlist.append(0.0)
            for inst in self.instances:
                numlist[inst.inst.get_integrated_label(m).val] += 1.0
            s = sum(numlist)
            for k in range(1, self.K + 1):
                self.theta_list[m][k].append(numlist[k] / s)
        for w in self.workers:
            w.compute_pi(self.instances)