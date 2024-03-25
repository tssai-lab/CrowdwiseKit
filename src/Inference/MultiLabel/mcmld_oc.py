# -*- coding: utf-8 -*-
# Multi-Class Multi-Label  Dependent One-Coin Model (MCMLD-OC)
# @author: Jing Zhang (jzhang@njust.edu.cn)

import numpy
import math
import random
from Inference.MultiLabel.core import data, samplable, utils
from Inference.MultiLabel import model

class MCMLDOCWorker:

    def __init__(self, worker):
        self.worker = worker
        self.M = 0
        self.K = 0
        self.R = 0
        self.rho_dict = dict()

    def initialize(self, M, K, R):
        self.M = M
        self.K = K
        self.R = R
        # initialize rho dictionary
        for r in range(1, self.R + 1):
            rho_list = [None]
            for m in range(1, self.M + 1):
                rho = samplable.RealV(0.0)
                self.random_initialize_rho(rho, 0.7, 0.9)
                rho_list.append(rho)
            self.rho_dict.setdefault(r,rho_list)
        #self.print_rhos()

    def random_initialize_rho(self, rho, diagonal_low, diagonal_high):
        val =  random.uniform(diagonal_low, diagonal_high)
        rho.setV(val)

    def m_update(self, instances, m):
        for r in range(1, self.R + 1):
            curr_rho = numpy.ndarray(shape=(self.K + 1), dtype=float, order='C')
            curr_rho.fill(0.0)
            total = 0.0
            for inst in instances:
                d = self.worker.get_label_val_for_inst(inst.inst.id, m)
                if d != 0:
                    for index in range(1, len(inst.y_combination)):
                        y_cmb = inst.y_combination[index]
                        if (d == y_cmb[m]):
                            curr_rho[d] += (inst.y_prob_list[index].getV() * inst.z_prob_list[r].getV())
                    total += (1.0 * inst.z_prob_list[r].getV())
            self.rho_dict.get(r)[m].append(sum(curr_rho) / total)

    def m_update_hard(self, instances, m):
        for r in range(1, self.R + 1):
            correct = 0.0
            total = 0.0
            for inst in instances:
                d = self.worker.get_label_val_for_inst(inst.inst.id, m)
                if d != 0:
                    if inst.y_list[m].getV() == d:
                        correct += 1.0
                    total += 1.0
            self.rho_dict.get(r)[m].append(correct/total)

    def print_rhos(self):
        for r in range(1, self.R + 1):
            for m in range(1, self.M + 1):
                print('rho (' + str(m) + ') of worker ' + str(self.worker.id) + ' in branch (' + str(r) + '):')
                self.rho_dict.get(r)[m].print_obj()
                print('')


class MCMLDOCInstance:

    def __init__(self, inst):
        self.inst = inst
        self.M = 0
        self.K = 0
        self.R = 0
        self.y_list = [None]
        self.y_prob_list = [None]
        self.likelihood = samplable.RealV(0.0)
        self.y_combination = [None]
        self.z = None
        self.z_prob_list = [None]

    def initialize(self, M, K, R):
        self.M = M
        self.K = K
        self.R = R
        for m in range(1, self.M + 1):
            self.y_list.append(samplable.IntV(0))
        self.y_combination.extend(utils.get_full_combination(self.M, self.K))
        for i in range(1, len(self.y_combination)):
            self.y_prob_list.append(samplable.RealV(0.0))
        self.z = samplable.IntV(0)
        for r in range (1, self.R + 1):
            self.z_prob_list.append(samplable.RealV(0.0))
        #self.print_ys()
        #self.print_y_probs()
        #self.print_z()

    def compute_likelihood(self, workers, theta_dict, omega):
        prob_L = []
        for index in range(1, len(self.y_combination)):
            y_cmb = self.y_combination[index]
            prod_theta = [0.0]
            prod_rho = [0.0]
            theta_rho = [0.0]
            for r in range(1, self.R + 1):
                prod_theta.append(1.0)
                prod_rho.append(1.0)
                theta_rho.append(1.0)
            for r in range(1, self.R + 1):
                theta_list = theta_dict.get(r)
                for m in range(1, self.M + 1):
                    prod_theta[r] *= theta_list[m][y_cmb[m]].getV()
                for w in workers:
                    for label_id in range(1, self.M + 1):
                        val = w.worker.get_label_val_for_inst(self.inst.id, label_id)
                        if val == 0:
                            continue
                        elif val == y_cmb[label_id]:
                            prod_rho[r] *= w.rho_dict.get(r)[label_id].getV()
                        else:
                            prod_rho[r] *= ((1.0 - w.rho_dict.get(r)[label_id].getV()) / (self.K - 1))
                theta_rho[r] *= (prod_theta[r] * prod_rho[r] * omega[r].getV())
            prob_L.append(sum(theta_rho))
        self.likelihood.append(sum(prob_L))
        return self.likelihood.getV()

    def e_step(self, workers, theta_dict, omega):
        sum_z = 0.0
        sum_y = 0.0
        z_prob = [None]
        for r in range(1, self.R + 1):
            z_prob.append(0.0)
        for index in range(1, len(self.y_combination)):
            y_cmb = self.y_combination[index]
            prod_theta = [0.0]
            prod_rho = [0.0]
            theta_rho = [0.0]
            for r in range(1, self.R + 1):
                prod_theta.append(1.0)
                prod_rho.append(1.0)
                theta_rho.append(1.0)
            for r in range(1, self.R + 1):
                theta_list = theta_dict.get(r)
                for m in range(1, self.M + 1):
                    prod_theta[r] *= theta_list[m][y_cmb[m]].getV()
                for w in workers:
                    for label_id in range(1, self.M + 1):
                        val = w.worker.get_label_val_for_inst(self.inst.id, label_id)
                        if val == 0:
                            continue
                        elif val == y_cmb[label_id]:
                            prod_rho[r] *= w.rho_dict.get(r)[label_id].getV()
                        else:
                            prod_rho[r] *= ((1.0 - w.rho_dict.get(r)[label_id].getV()) / (self.K - 1))
                theta_rho[r] *= (prod_theta[r] * prod_rho[r] * omega[r].getV())
                z_prob[r] += theta_rho[r]
                sum_z += theta_rho[r]
            self.y_prob_list[index].append(sum(theta_rho))
            sum_y += self.y_prob_list[index].getV()
        # uniform
        for r in range(1, self.R + 1):
            self.z_prob_list[r].append(z_prob[r] / sum_z)
        for index in range(1, len(self.y_combination)):
            self.y_prob_list[index].setV(self.y_prob_list[index].getV() / sum_y)

    def e_step_hard(self, workers, theta_dict, omega):
        sum_z = 0.0
        z_prob = [None]
        for r in range(1, self.R + 1):
            z_prob.append(0.0)
        for index in range(1, len(self.y_combination)):
            y_cmb = self.y_combination[index]
            prod_theta = [0.0]
            prod_rho = [0.0]
            theta_rho = [0.0]
            for r in range(1, self.R + 1):
                prod_theta.append(1.0)
                prod_rho.append(1.0)
                theta_rho.append(1.0)
            for r in range(1, self.R + 1):
                theta_list = theta_dict.get(r)
                for m in range(1, self.M + 1):
                    prod_theta[r] *= theta_list[m][y_cmb[m]].getV()
                for w in workers:
                    for label_id in range(1, self.M + 1):
                        val = w.worker.get_label_val_for_inst(self.inst.id, label_id)
                        if val == 0:
                            continue
                        elif val == y_cmb[label_id]:
                            prod_rho[r] *= w.rho_dict.get(r)[label_id].getV()
                        else:
                            prod_rho[r] *= ((1.0 - w.rho_dict.get(r)[label_id].getV()) / (self.K - 1))
                theta_rho[r] *= (prod_theta[r] * prod_rho[r] * omega[r].getV())
                z_prob[r] += theta_rho[r]
                sum_z += theta_rho[r]
            self.y_prob_list[index].append(sum(theta_rho))
        for r in range(1, self.R + 1):
            self.z_prob_list[r].append(z_prob[r]/sum_z)
        estimated_y_pos = self.compute_estimated_ys()
        for m in range(1, self.M + 1):
            self.y_list[m].append(self.y_combination[estimated_y_pos][m])
        estimated_z_pos = self.compute_estimated_z()
        self.z.append(estimated_z_pos)

    def compute_estimated_ys(self, random=False):
        prob = []
        prob.append(None)
        for index in range(1, len(self.y_prob_list)):
            prob.append(self.y_prob_list[index].getV())
        maxindex = utils.get_max_index(prob, random)
        return maxindex

    def compute_estimated_z(self, random=False):
        prob = []
        prob.append(None)
        for index in range(1, len(self.z_prob_list)):
            prob.append(self.z_prob_list[index].getV())
        maxindex = utils.get_max_index(prob, random)
        return maxindex

    def final_aggregate(self, soft=True):
        if (soft == True):
            estimated_pos = self.compute_estimated_ys(True)
            for m in range(1, self.M + 1):
                self.y_list[m].append(self.y_combination[estimated_pos][m])
            estimated_z_pos = self.compute_estimated_z(True)
            self.z.append(estimated_z_pos)
        #print('instance ' + str(self.inst.id) + ' get z = ' + str(self.z.getV()))
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

    def print_z(self):
        print('The estimated cluster of instance ' + str(self.inst.id) + ':', end=' ')
        self.z.print_obj()
        print('with probabilities ' + str(self.z_prob_list), end=' ')
        print('')


class MCMLDOCModel(model.Model):
    """
    dependent One-Coin model
    """
    def __init__(self, R, maxrnd):
        model.Model.__init__(self)
        self.workers = []
        self.instances = []
        self.I = 0
        self.J = 0
        self.K = 0
        self.M = 0  # M is the number of label ids
        self.R = R
        self.theta_dict = dict()
        self.maxround = maxrnd
        self.omega = [None]

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
            mcmldoc_inst = MCMLDOCInstance(inst)
            mcmldoc_inst.initialize(self.M, self.K, self.R)
            self.instances.append(mcmldoc_inst)
        for w in range(1, self.J + 1):
            worker = dataset.get_worker(w)
            mcmldoc_worker = MCMLDOCWorker(worker)
            mcmldoc_worker.initialize(self.M, self.K, self.R)
            self.workers.append(mcmldoc_worker)
        for r in  range(1, self.R + 1):
            theta_list = [None]
            for m in range(1, self.M + 1):
                thetas = numpy.ndarray(shape=(self.K + 1), dtype=samplable.RealV, order='C')
                vals =  utils.gen_rand_sum_one(self.K) # todo: can optimize
                for i in range(1, self.K + 1):
                    thetas[i] = samplable.RealV(vals[i-1])
                theta_list.append(thetas)
            self.theta_dict.setdefault(r, theta_list)
        # initialize omega
        if (len(self.omega)==1):
            vals = utils.gen_uniform_sum_one(self.R)
            for r in range(1, self.R + 1):
                self.omega.append(samplable.RealV(vals[r - 1]))

    def set_omega(self, vals):
        for r in range(1, self.R + 1):
            self.omega.append(samplable.RealV(vals[r]))

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
            l = inst.compute_likelihood(self.workers, self.theta_dict, self.omega)
            log_like += math.log(l)
        return log_like

    def e_step(self):
        for inst in self.instances:
            inst.e_step(self.workers, self.theta_dict, self.omega)

    def e_step_hard(self):
        for inst in self.instances:
            inst.e_step_hard(self.workers, self.theta_dict, self.omega)

    def m_step(self):
        # calculate omega
        num_inst = []
        for r in range(0, self.R + 1):
            num_inst.append(0.0)
        for inst in self.instances:
            for r in range(1, self.R + 1):
                num_inst[r] += inst.z_prob_list[r].getV()
        s = sum(num_inst)
        for r in range(1, self.R + 1):
            self.omega[r].append(num_inst[r] / s)
            #print('w[' + str(r) + ']=' + str(self.omega[r].getV()))
        for m in range(1, self.M + 1):
            self.m_update(m)

    def m_step_hard(self):
        # calculate omega
        num_inst = []
        for r in range(0, self.R + 1):
            num_inst.append(0.0)
        for inst in self.instances:
            num_inst[inst.z.getV()] += 1.0
        s = sum(num_inst)
        for r in range(1, self.R + 1):
            self.omega[r].append(num_inst[r] / s)
            #print('w[' + str(r) + ']=' + str(self.omega[r].getV()))
        for m in range(1, self.M + 1):
            self.m_update_hard(m)

    def m_update(self, m):
        rk = numpy.ndarray(shape=(self.R + 1, self.K + 1), dtype=float, order='C')
        rk.fill(0.0)
        for inst in self.instances:
            for index in range(1, len(inst.y_combination)):
                y_cmb = inst.y_combination[index]
                for r in range(1, self.R + 1):
                    rk[r][y_cmb[m]] += (inst.y_prob_list[index].getV() * inst.z_prob_list[r].getV())
        sumr = numpy.ndarray(shape=(self.R + 1), dtype=float, order='C')
        sumr.fill(0.0)
        for r in range(1, self.R + 1):
            for k in range(1, self.K + 1):
                sumr[r] += rk[r][k]
        for r in range(1, self.R + 1):
            for k in range(1, self.K + 1):
                if (sumr[r] == 0.0):
                    self.theta_dict.get(r)[m][k].append(0.0)
                else:
                    self.theta_dict.get(r)[m][k].append(rk[r][k] / sumr[r])
        # self.print_theta()
        for w in self.workers:
            w.m_update(self.instances, m)

    def m_update_hard(self, m):
        # calculate theta
        for r in range(1, self.R + 1):
            numlist = []
            for k in range(0, self.K + 1):
                numlist.append(0.0)
            for inst in self.instances:
                if inst.z.getV() == r:
                    numlist[inst.y_list[m].getV()] += 1.0
            ss = sum(numlist)
            for k in range(1, self.K + 1):
                if (ss == 0):
                    self.theta_dict.get(r)[m][k].append(0.0)
                else:
                    self.theta_dict.get(r)[m][k].append(numlist[k] / ss)
        # self.print_theta()
        for w in self.workers:
            w.m_update_hard(self.instances, m)

    def infer(self, dataset, soft=True):
        self.initialize(dataset)
        count = 1
        last_likelihood = -999999999
        curr_likehihood = self.loglikelihood()
        print('MCMLDOC initial log-likelihood = ' + str(curr_likehihood))
        while ((count <= self.maxround) and (abs(curr_likehihood - last_likelihood) / abs(last_likelihood) > self.converge_rate)):
            if (soft == True):
                self.e_step()
                self.m_step()
            else:
                self.e_step_hard()
                self.m_step_hard()
            last_likelihood = curr_likehihood
            curr_likehihood = self.loglikelihood()
            print('MCMLDOC round (' + str(count) + ') log-likelihood = ' + str(curr_likehihood))
            count += 1
        self.final_aggregate(soft)

    def final_aggregate(self, soft):
        for inst in self.instances:
            inst.final_aggregate(soft)

    def print_theta(self):
        for m in range(1, self.M+1):
            print('probability of classes on label (' + str(m) + '):')
            for r in range(1, self.R+ 1):
                for k in range(1, self.K + 1):
                    self.theta_dict.get(r)[m][k].print_obj()
                print('')