# -*- coding: utf-8 -*-
# Multi-Class Multi-Label Independent One-Coin Model (MCMLI-OC)
# @author: Jing Zhang (jzhang@njust.edu.cn)

import numpy
import math
import random
from Inference.MultiLabel.core import data, samplable, utils
from Inference.MultiLabel import model

class MCMLIOCWorker:

    def __init__(self, worker):
        self.worker = worker
        self.M = 0
        self.K = 0
        self.rho_list = list()

    def initialize(self, M, K):
        self.M = M
        self.K = K
        self.rho_list.append(None)
        for m in range(1, self.M + 1):
            rho = samplable.RealV(0.0)
            self.random_initialize_rho(rho, 0.7, 0.9)
            self.rho_list.append(rho)
        #self.print_rhos()

    def random_initialize_rho(self, rho, diagonal_low, diagonal_high):
        val =  random.uniform(diagonal_low, diagonal_high)
        rho.setV(val)

    def m_update(self, instances, m):
        curr_rho = numpy.ndarray(shape=(self.K + 1), dtype=float, order='C')
        curr_rho.fill(0.0)
        total = 0.0
        for inst in instances:
            d = self.worker.get_label_val_for_inst(inst.inst.id, m)
            if d != 0:
                for index in range(1, len(inst.y_combination)):
                    y_cmb = inst.y_combination[index]
                    if (d == y_cmb[m]):
                        curr_rho[d] += inst.y_prob_list[index].getV()
                total += 1.0
        self.rho_list[m].append(sum(curr_rho) / total)

    def m_update_hard(self, instances, m):
        correct = 0.0
        total = 0.0
        for inst in instances:
            d = self.worker.get_label_val_for_inst(inst.inst.id, m)
            if d != 0:
                if inst.y_list[m].getV() == d:
                    correct += 1.0
                total += 1.0
        self.rho_list[m].append(correct/total)

    def print_rhos(self):
        for m in range(1, self.M + 1):
            print('rho (' + str(m) + ') of worker ' + str(self.worker.id) + '):')
            self.rho_list[m].print_obj()
            print('')


class MCMLIOCInstance:

    def __init__(self, inst):
        self.inst = inst
        self.M = 0
        self.K = 0
        self.y_list = [None]
        self.y_prob_list = [None]
        self.likelihood = samplable.RealV(0.0)
        self.y_combination = [None]

    def initialize(self, M, K):
        self.M = M
        self.K = K
        for m in range(1, self.M + 1):
            self.y_list.append(samplable.IntV(0))
        self.y_combination.extend(utils.get_full_combination(self.M, self.K))
        for i in range(1, len(self.y_combination)):
            self.y_prob_list.append(samplable.RealV(0.0))
        #self.print_ys()
        #self.print_y_probs()

    def compute_likelihood(self, workers, theta_list):
        prob_L = []
        for index in range(1, len(self.y_combination)):
            y_cmb = self.y_combination[index]
            prod_theta = 1.0
            for m in range(1, self.M + 1):
                prod_theta *= theta_list[m][y_cmb[m]].getV()
            prod_rho = 1.0
            for w in workers:
                for label_id in range(1, self.M + 1):
                    val = w.worker.get_label_val_for_inst(self.inst.id, label_id)
                    if val == 0:
                        continue
                    elif val == y_cmb[label_id]:
                        prod_rho *= w.rho_list[label_id].getV()
                    else:
                        prod_rho *= ((1.0 - w.rho_list[label_id].getV()) / (self.K - 1))
            prob_L.append(prod_theta * prod_rho)
        self.likelihood.append(sum(prob_L))
        return self.likelihood.getV()

    def e_step(self, workers, theta_list):
        sum = 0.0
        for index in range(1, len(self.y_combination)):
            y_cmb = self.y_combination[index]
            prod_theta = 1.0
            for m in range(1, self.M + 1):
                prod_theta *= theta_list[m][y_cmb[m]].getV()
            prod_rho = 1.0
            for w in workers:
                for label_id in range(1, self.M + 1):
                    val = w.worker.get_label_val_for_inst(self.inst.id, label_id)
                    if val == 0:
                        continue
                    elif val == y_cmb[label_id]:
                        prod_rho *= w.rho_list[label_id].getV()
                    else:
                        prod_rho *= ((1.0 - w.rho_list[label_id].getV()) / (self.K - 1))
            self.y_prob_list[index].append(prod_theta * prod_rho)
            sum += self.y_prob_list[index].getV()
        # uniform
        for index in range(1, len(self.y_combination)):
            self.y_prob_list[index].setV(self.y_prob_list[index].getV() / sum)

    def e_step_hard(self, workers, theta_list):
        for index in range(1, len(self.y_combination)):
            y_cmb = self.y_combination[index]
            prod_theta = 1.0
            for m in range(1, self.M + 1):
                prod_theta *= theta_list[m][y_cmb[m]].getV()
            prod_rho = 1.0
            for w in workers:
                for label_id in range(1, self.M + 1):
                    val = w.worker.get_label_val_for_inst(self.inst.id, label_id)
                    if val == 0:
                        continue
                    elif val == y_cmb[label_id]:
                        prod_rho *= w.rho_list[label_id].getV()
                    else:
                        prod_rho *= ((1.0 - w.rho_list[label_id].getV()) / (self.K - 1))
            self.y_prob_list[index].append(prod_theta * prod_rho)
        estimated_y_pos = self.compute_estimated_ys()
        for m in range(1, self.M + 1):
            self.y_list[m].append(self.y_combination[estimated_y_pos][m])

    def compute_estimated_ys(self, random=False):
        prob = []
        prob.append(None)
        for index in range(1, len(self.y_prob_list)):
            prob.append(self.y_prob_list[index].getV())
        maxindex = utils.get_max_index(prob, random)
        return maxindex

    def final_aggregate(self, soft=True):
        if (soft == True):
            estimated_pos = self.compute_estimated_ys(True)
            for m in range(1, self.M + 1):
                self.y_list[m].append(self.y_combination[estimated_pos][m])
        for m in range(1, self.M + 1):
            y_val = self.y_list[m].getV()
            # set integrated label
            integrated_label = data.Label(m)
            integrated_label.inst_id = self.inst.id
            integrated_label.worker_id = data.Worker.AGGR
            integrated_label.val = y_val
            self.inst.add_integrated_label(integrated_label)
            # print('instance ' + str(self.inst.id) + ' get integrated label (' + str(m) + ') :' + str(integrated_label.val))

    def print_ys(self):
        print('The estimated class of instance ' + str(self.inst.id) + ':', end=' ')
        for m in range(1, self.M + 1):
            self.y_list[m].print_obj()
        print('')

    def print_y_probs(self):
        print('The probabilities of classes of instance ' + str(self.inst.id) + ' >>>')
        for m in range(1, self.M + 1):
            print('label (' + str(m) + '):', end=' ')
            for k in range(1, self.K + 1):
                self.y_prob_list[m][k].print_obj()
            print('')


class MCMLIOCModel(model.Model):
    """
    independent One-Coin model
    """
    def __init__(self, maxrnd):
        model.Model.__init__(self)
        self.workers = []
        self.instances = []
        self.I = 0
        self.J = 0
        self.K = 0
        self.M = 0  # M is the number of label ids
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
            mcmlioc_inst = MCMLIOCInstance(inst)
            mcmlioc_inst.initialize(self.M, self.K)
            self.instances.append(mcmlioc_inst)
        for w in range(1, self.J + 1):
            worker = dataset.get_worker(w)
            mcmlioc_worker = MCMLIOCWorker(worker)
            mcmlioc_worker.initialize(self.M, self.K)
            self.workers.append(mcmlioc_worker)
        for m in range(1, self.M + 1):
            thetas = numpy.ndarray(shape=(self.K + 1), dtype=samplable.RealV, order='C')
            for i in range(1, self.K + 1):
                thetas[i] = samplable.RealV(1.0 / self.K)
            self.theta_list.append(thetas)

    def initialK(self, dataset):
        maxK = 0
        for k in range(1, self.M + 1):
            currentK = dataset.get_label_val_size(k)
            if currentK > maxK:
                maxK = currentK
        self.K = maxK

    def loglikelihood(self):
        log_like = 0
        for inst in self.instances:
            l = inst.compute_likelihood(self.workers, self.theta_list)
            log_like += math.log(l)
        return log_like

    def e_step(self):
        for inst in self.instances:
            inst.e_step(self.workers, self.theta_list)

    def e_step_hard(self):
        for inst in self.instances:
            inst.e_step_hard(self.workers, self.theta_list)

    def m_step(self):
        for m in range(1, self.M + 1):
            self.m_update(m)

    def m_step_hard(self):
        for m in range(1, self.M + 1):
            self.m_update_hard(m)

    def m_update(self, m):
        # calculate theta
        numlist = []
        for k in range(0, self.K + 1):
            numlist.append(0.0)
        for inst in self.instances:
            for index in range(1, len(inst.y_combination)):
                y_cmb = inst.y_combination[index]
                numlist[y_cmb[m]] += inst.y_prob_list[index].getV()
        s = sum(numlist)
        for k in range(1, self.K + 1):
            self.theta_list[m][k].append(numlist[k] / s)
        # self.print_theta()
        for w in self.workers:
            w.m_update(self.instances, m)

    def m_update_hard(self, m):
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
            w.m_update_hard(self.instances, m)

    def infer(self, dataset, soft=True):
        self.initialize(dataset)
        count = 1
        last_likelihood = -999999999
        curr_likehihood = self.loglikelihood()
        print('MCMLIOC initial log-likelihood = ' + str(curr_likehihood))
        while ((count <= self.maxround) and (abs(curr_likehihood - last_likelihood) / abs(last_likelihood) > self.converge_rate)):
            if (soft == True):
                self.e_step()
                self.m_step()
            else:
                self.e_step_hard()
                self.m_step_hard()
            last_likelihood = curr_likehihood
            curr_likehihood = self.loglikelihood()
            print('MCMLIOC round (' + str(count) + ') log-likelihood = ' + str(curr_likehihood))
            count += 1
        self.final_aggregate(soft)

    def final_aggregate(self, soft):
        for inst in self.instances:
            inst.final_aggregate(soft)

    def print_theta(self):
        for m in range(1, self.M + 1):
            print('probability of classes on label (' + str(m) +'):')
            for k in range(1, self.K + 1):
                self.theta_list[m][k].print_obj()
            print('')