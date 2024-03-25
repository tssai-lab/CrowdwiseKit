# -*- coding: utf-8 -*-
# Multi-Class One-Coin model (MCOC)
# @author: Jing Zhang (jzhang@njust.edu.cn)

import numpy
import math
from Inference.MultiLabel.core import data, samplable, utils
from Inference.MultiLabel import model

class MCOCWorker:

    def __init__(self, worker):
        self.worker = worker
        self.M = 0
        self.K = 0
        self.rho_list = []

    def initialize(self, M, K):
        self.M = M
        self.K = K
        self.rho_list.append(None)
        for m in range(1, self.M + 1):
            rho = samplable.RealV(0.0)
            self.initialize_rho(rho)
            self.rho_list.append(rho)
        #self.print_rhos()

    def initialize_rho(self,  rho):
        rho.setV(0.8)

    def m_step(self, instances, m):
        curr_rho = numpy.ndarray(shape=(self.K + 1), dtype=float, order='C')
        curr_rho.fill(0.0)
        total = 0.0
        for inst in instances:
            d = self.worker.get_label_val_for_inst(inst.inst.id, m)
            if (d != 0):
                for k in range(1, self.K + 1):
                    if (d == k):
                        curr_rho[d] += inst.y_prob_list[m][k].getV()
                total += 1.0
        self.rho_list[m].append(sum(curr_rho) / total)

    def m_step_hard(self, instances, m):
        correct = 0.0
        total = 0.0
        for inst in instances:
            d = self.worker.get_label_val_for_inst(inst.inst.id, m)
            if (d != 0):
                if inst.y_list[m].getV() == d:
                    correct += 1.0
                total += 1.0
        self.rho_list[m].append(correct/total)

    def print_rho(self):
        for m in range(1, self.M + 1):
            print('rho (' + str(m) + ') of worker ' + str(self.worker.id) + ':')
            self.rho_list[m].print_obj()
            print('')


class MCOCInstance:

    def __init__(self, inst):
        self.inst = inst
        self.M = 0
        self.K = 0
        self.y_list = [None]
        self.y_prob_list = [None]
        self.likelihood_list = [None]

    def initialize(self, M, K):
        self.M = M
        self.K = K
        for m in range(1, self.M + 1):
            self.y_list.append(samplable.IntV(0))
            probs = numpy.ndarray(shape=(self.K + 1), dtype=samplable.RealV, order='C')
            for i in range(1, self.K + 1):
                probs[i] = samplable.RealV(1.0 / self.K)
            self.y_prob_list.append(probs)
            self.likelihood_list.append(samplable.RealV(0.0))
        #self.print_ys()
        #self.print_y_probs()

    def compute_likelihood(self, workers, thetas, m):
        p_k = []
        for k in range(1, self.K + 1):
            prod = 1.0
            for w in workers:
                val = w.worker.get_label_val_for_inst(self.inst.id, m)
                if val == 0:
                    continue
                elif val == k:
                    prod *= w.rho_list[m].getV()
                else:
                    prod *= ((1.0 - w.rho_list[m].getV())/(self.K - 1))
            p_k.append(prod * thetas[k].getV())
        self.likelihood_list[m].append(sum(p_k))
        return self.likelihood_list[m].getV()

    def e_step(self, workers, thetas, m):
        sum = 0.0
        for k in range(1, self.K + 1):
            prod = 1.0
            for w in workers:
                val = w.worker.get_label_val_for_inst(self.inst.id, m)
                if val == 0:
                    continue
                elif val == k:
                    prod *= w.rho_list[m].getV()
                else:
                    prod *= ((1.0 - w.rho_list[m].getV()) / (self.K - 1))
            self.y_prob_list[m][k].append(prod * thetas[k].getV())
            sum += self.y_prob_list[m][k].getV()
        # uniform
        for k in range(1, self.K + 1):
            self.y_prob_list[m][k].setV(self.y_prob_list[m][k].getV() / sum)

    def e_step_hard(self, workers, thetas, m):
        for k in range(1, self.K + 1):
            prod = 1.0
            for w in workers:
                val = w.worker.get_label_val_for_inst(self.inst.id, m)
                if val == 0:
                    continue
                elif val == k:
                    prod *= w.rho_list[m].getV()
                else:
                    prod *= ((1.0 - w.rho_list[m].getV()) / (self.K - 1))
            self.y_prob_list[m][k].append(prod * thetas[k].getV())
        estimated_y = self.compute_estimated_y(m)
        self.y_list[m].append(estimated_y)
        # print('inst '+ str(self.inst.id) + ' get estimated y(' + str(m) + ') = '+ str(estimated_y))

    def compute_estimated_y(self, m, random=False):
        prob = []
        prob.append(None)
        for k in range(1, self.K + 1):
            prob.append(self.y_prob_list[m][k].getV())
        maxindex = utils.get_max_index(prob, random)
        return maxindex

    def final_aggregate(self, m, soft=True):
        if (soft == True):
            estimated_y = self.compute_estimated_y(m, True)
            self.y_list[m].append(estimated_y)
        y_val = self.y_list[m].getV()
        # set integrated label
        integrated_label = data.Label(m)
        integrated_label.inst_id = self.inst.id
        integrated_label.worker_id = data.Worker.AGGR
        integrated_label.val = y_val
        self.inst.add_integrated_label(integrated_label)
        #print('instance ' + str(self.inst.id) + ' get integrated label (' + str(m) + ') :' + str(integrated_label.val))

    def print_ys(self):
        print('The estimated class of instance ' + str(self.inst.id) +  ':', end=' ')
        for m in range(1, self.M + 1):
            self.y_list[m].print_obj()
        print('')

    def print_y_probs(self):
        print('The probabilities of classes of instance ' + str(self.inst.id) + ' >>>')
        for m in range(1, self.M + 1):
            print('label (' + str(m)+ '):', end=' ')
            for k in range(1, self.K + 1):
                self.y_prob_list[m][k].print_obj()
            print('')


class MCOCModel(model.Model):
    """
    One-Coin model for multi-class labeling model
    """
    def __init__(self, maxrnd):
        model.Model.__init__(self)
        self.workers = []
        self.instances = []
        self.I = 0
        self.J = 0
        self.K = 0
        self.M = 0 # M is the number of label ids
        self.theta_list = [None]
        self.maxround = maxrnd

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
            mcoc_inst = MCOCInstance(inst)
            mcoc_inst.initialize(self.M, self.K)
            self.instances.append(mcoc_inst)
        for w in range(1, self.J + 1):
            worker = dataset.get_worker(w)
            mcoc_worker = MCOCWorker(worker)
            mcoc_worker.initialize(self.M, self.K)
            self.workers.append(mcoc_worker)
        for m in range(1, self.M + 1):
            thetas = numpy.ndarray(shape=(self.K+1), dtype=samplable.RealV, order='C')
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

    def loglikelihood(self, m):
        log_like = 0
        for inst in self.instances:
            l = inst.compute_likelihood(self.workers, self.theta_list[m], m)
            log_like += math.log(l)
        return log_like

    def e_step(self, m):
        for inst in self.instances:
            inst.e_step(self.workers, self.theta_list[m], m)

    def e_step_hard(self, m):
        for inst in self.instances:
            inst.e_step_hard(self.workers, self.theta_list[m], m)

    def m_step(self, m):
        # calculate theta
        numlist = []
        for k in range(0, self.K + 1):
            numlist.append(0.0)
        for inst in self.instances:
            for k in range(1, self.K + 1):
                numlist[k] += inst.y_prob_list[m][k].getV()
        s = sum(numlist)
        for k in range(1, self.K + 1):
            self.theta_list[m][k].append(numlist[k] / s)
        # self.print_theta()
        for w in self.workers:
            w.m_step(self.instances, m)

    def m_step_hard(self, m):
        # calculate theta
        numlist = []
        for k in range(0, self.K + 1):
            numlist.append(0.0)
        for inst in self.instances:
            numlist[inst.y_list[m].getV()] += 1.0
        s = sum(numlist)
        for k in range(1, self.K + 1):
            self.theta_list[m][k].append(numlist[k] / s)
        # self.print_theta()
        for w in self.workers:
            w.m_step_hard(self.instances, m)

    def em(self, m, soft=True):
        count = 1
        last_likelihood = -999999999
        curr_likehihood = self.loglikelihood(m)
        print('MCOC on label (' + str(m) + ') initial log-likelihood = ' + str(curr_likehihood))
        while ((count <= self.maxround) and (abs(curr_likehihood - last_likelihood) / abs(last_likelihood) > self.converge_rate)):
            if (soft == True):
                self.e_step(m)
                self.m_step(m)
            else:
                self.e_step_hard(m)
                self.m_step_hard(m)
            last_likelihood = curr_likehihood
            curr_likehihood = self.loglikelihood(m)
            print('MCOC on label (' + str(m) + ') round (' + str(count) +') log-likelihood = ' + str(curr_likehihood))
            count += 1

    def final_aggregate(self, m, soft):
        for inst in self.instances:
            inst.final_aggregate(m, soft)

    def infer(self, dataset, soft=True):
        self.initialize(dataset)
        for m in range(1, self.M + 1):
            self.em(m, soft)
            self.final_aggregate(m, soft)

    def print_theta(self):
        for m in range(1, self.M + 1):
            print('probability of classes on label (' + str(m) + '):')
            for k in range(1, self.K + 1):
                self.theta_list[m][k].print_obj()
            print('')