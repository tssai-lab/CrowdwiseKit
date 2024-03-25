# -*- coding: UTF-8 -*-
# Independent Bayesian Classifier Combination (iBCC)
# Reference: Kim, H. C., & Ghahramani, Z. (2012, March). Bayesian classifier combination.
#            In Artificial Intelligence and Statistics (pp. 619-627).
# @author: Jing Zhang (jzhang@njust.edu.cn)

import numpy
from Inference.MultiLabel.core import data, samplable, perf, utils
from Inference.MultiLabel import model

class IBCCWorker:

    def __init__(self, worker):
        self.worker = worker
        self.K = 0
        self.M = 0
        self.pi_list = [None]
        self.psi_list = [None]

    def initialize(self, M, K):
        self.M = M
        self.K = K
        for m in range(1, self.M + 1):
            pi = numpy.ndarray(shape=(self.K + 1, self.K + 1), dtype=samplable.RealV, order='C')
            psi = numpy.ndarray(shape=(self.K + 1, self.K + 1), dtype=samplable.RealV, order='C')
            self.initialize_pi_psi(pi, psi)
            self.pi_list.append(pi)
            self.psi_list.append(psi)

    def initialize_pi_psi(self, pi, psi):
        # initialize pi
        for i in range(1, self.K + 1):
            for j in range(1, self.K + 1):
              ##  self.pi[i][j].setVal(1.0/self.K)
                if i == j:
                    pi[i][j] = samplable.RealV(0.80)
                else:
                    pi[i][j] = samplable.RealV(0.20/(self.K - 1))
                psi[i][j] = samplable.RealV(1.0)

    def print_pi(self, m, round = 0):
        print('pi of worker ' + str(self.worker.id) + 'on label (' +str(m) +') in round ' + str(round) + ':')
        for i in range(1, self.K + 1):
            for j in range(1, self.K + 1):
                self.pi_list[m][i][j].print_obj(round)
            print('')

    def print_psi(self, m, round = 0):
        print('psi of worker ' + str(self.worker.id) + 'on label (' +str(m) +')  in round ' + str(round) + ':')
        for i in range(1, self.K + 1):
            for j in range(1, self.K + 1):
                self.psi_list[m][i][j].print_obj(round)
            print('')

    def sampling_pi(self, label_id, round, instances):
        for k in range(1, self.K+1):
            #sampling pi_k.
            # calculate alphas for dirichlet distribution
            alpha = []
            for d in range(1, self.K + 1):
                num_of_kd = 0
                for inst in instances:
                    if (inst.y_list[label_id].getV() == k) and (self.worker.get_label_val_for_inst(inst.inst.id, label_id)==d):
                        num_of_kd += 1
                alpha.append(num_of_kd+self.psi_list[label_id][k][d].getV())
            pi_k_new = numpy.random.dirichlet(alpha)
            # update prob
            for d in range(1, self.K + 1):
                self.pi_list[label_id][k][d].append(pi_k_new[d - 1])


class IBCCInstance:

    def __init__(self, instance):
        self.inst = instance
        self.M = 0
        self.K = 0
        self.y_list = [None]
        self.y_prob_list = [None]

    def initialize(self, M, K):
        self.M = M
        self.K = K
        for m in range(1, self.M + 1):
            self.y_list.append(samplable.IntV(0))
            probs = numpy.ndarray(shape=(self.K + 1), dtype=samplable.RealV, order='C')
            r = [numpy.random.random() for i in range(0, self.K)]
            s = sum(r)
            r = [i / s for i in r]
            for i in range(1, self.K + 1):
                probs[i] = samplable.RealV(r[i - 1])
            self.y_prob_list.append(probs)

    def compute_estimated_y(self, m, random=False):
        prob = []
        prob.append(None)
        for k in range(1, self.K + 1):
            prob.append(self.y_prob_list[m][k].getV())
        maxindex = utils.get_max_index(prob, random)
        return maxindex

    def print_y(self, m, round):
        print('The estimated class of instance ' + str(self.inst.id) +
              'on label (' +str(m)+ ') in round (' + str(round)+'):', end=' ')
        self.y_list[m].print_obj(round)
        print('')

    def print_y_probs(self, m, round):
        print('The probabilities of classes of instance ' + str(self.inst.id) +
              'on label (' + str(m) + '):')
        for k in range(1, self.K + 1):
            self.y_prob_list[m][k].print_obj()
        print('')

    def sampling_y(self, label_id, round, worker_list, prob):
        multinomial_pars = []
        for k in range(1, self.K + 1):
            prod = 1.0
            for w in worker_list:
                labelval = w.worker.get_label_val_for_inst(self.inst.id, label_id)
                if labelval == 0:
                    continue
                else:
                    prod *= w.pi_list[label_id][k][labelval].getV()
            prod *= prob[k].getV()
            multinomial_pars.append(prod)
        s = sum(multinomial_pars)
        multinomial_pars = [i / s for i in multinomial_pars]
        #update yp
        for k in range(1, self.K + 1):
            self.y_prob_list[label_id][k].append(multinomial_pars[k-1])
        #sample y
        y_k = numpy.random.multinomial(1, multinomial_pars)
        for pos in range(0, self.K):
            if y_k[pos] == 1:
                self.y_list[label_id].append(pos+1)

    def final_aggregate(self, m, begin):
        max_y = self.y_list[m].get_max_type(begin)
        # set integrated label
        integrated_label = data.Label(m)
        integrated_label.inst_id = self.inst.id
        integrated_label.worker_id = data.Worker.AGGR
        integrated_label.val = max_y
        self.inst.add_integrated_label(integrated_label)
        # print('instance ' + str(self.inst.id) + ' get integrated label (' + str(m) + ') :' + str(integrated_label.val))


class IBCCModel(model.Model):

    def __init__(self):
        model.Model.__init__(self)
        self.workers = []
        self.instances = []
        self.I = 0
        self.J = 0
        self.K = 0
        self.M = 0  # M is the number of label ids
        self.lmd_list = [None]
        self.prob_list = [None]
        self.nu_list = [None]

    def initialize(self, dataset):
        self.M = dataset.get_label_id_size()
        print('The number of label ids = ' + str(self.M))
        self.I = dataset.get_instance_size()
        self.J = dataset.get_worker_size()
        print('Total ' + str(self.I) + ' instances and ' + str(self.J) + ' workers')
        self.initialK(dataset)
        print('K = ' + str(self.K))
        for m in range(1, self.M + 1):
            lmd = numpy.ndarray(shape=(self.K + 1, self.K + 1), dtype=samplable.RealV, order='C')
            prob = [None]
            nu = [None]
            for i in range(1, self.K + 1):
                nu.append(samplable.RealV(1.0))
                # \lmb diagonals bigger than the off-diagonals
                for j in range(1, self.K + 1):
                    if i == j:
                        lmd[i][j] = samplable.RealV(1.5)
                    else:
                        lmd[i][j] = samplable.RealV(0.5)
                prob.append(samplable.RealV(1.0/self.K))
            self.nu_list.append(nu)
            self.lmd_list.append(lmd)
            self.prob_list.append(prob)
        for i in range(1, self.I + 1):
            inst = dataset.get_instance(i)
            ibcc_inst = IBCCInstance(inst)
            ibcc_inst.initialize(self.M, self.K)
            self.instances.append(ibcc_inst)
        for w in range(1, self.J + 1):
            worker = dataset.get_worker(w)
            ibcc_worker = IBCCWorker(worker)
            ibcc_worker.initialize(self.M, self.K)
            self.workers.append(ibcc_worker)

    def initialK(self, dataset):
        maxK = 0
        for k in range(1, self.M + 1):
            currentK = dataset.get_label_val_size(k)
            if currentK > maxK:
                maxK = currentK
        self.K = maxK

    def print_lmd(self, m, round=0):
        print('lambda on label (' +str(m)+ ') in round ' + str(round) + ':')
        for i in range(1, self.K + 1):
            for j in range(1, self.K + 1):
                self.lmd_list[m][i][j].print_obj(round)
            print('')

    def print_prob(self, m, round=0):
        print('probability of classes on label (' +str(m)+ ') in round ' + str(round) + ':')
        for i in range(1, self.K + 1):
            self.prob_list[m][i].print_obj(round)
        print('')

    def print_nu(self, m, round=0):
        print('nu on label (' +str(m)+ ') in round ' + str(round) + ':')
        for i in range(1, self.K + 1):
            self.nu_list[m][i].print_obj(round)
        print('')

    def sampling_prob(self, m, round):
        # calculate alphas for dirichlet distribution
        alpha = []
        for k in range(1, self.K+1):
            num_of_y_k = 0
            for inst in self.instances:
                if inst.y_list[m].getV() == k:
                    num_of_y_k += 1
            alpha.append(num_of_y_k+self.nu_list[m][k].getV())

        prob_new = numpy.random.dirichlet(alpha)
        #update prob
        for k in range(1, self.K + 1):
            self.prob_list[m][k].append(prob_new[k-1])

    def sampling_pi(self, m, round):
        for w in self.workers:
            w.sampling_pi(m, round, self.instances)

    def sampling_y(self, m, round):
        for inst in self.instances:
            inst.sampling_y(m, round, self.workers, self.prob_list[m])

    def sampling_psi(self, m, round):
        pass

    def final_aggregate(self, m, begin):
        for inst in self.instances:
            inst.final_aggregate(m, begin)

    def sampling_infer(self, dataset, maxround, begin):
        self.initialize(dataset)
        for m in range (1, self.M + 1):
            for r in range(1, maxround + 1):
                self.sampling_y(m, r)
                self.sampling_prob(m, r)
                self.sampling_pi(m, r)
                self.sampling_psi(m, r)
            print('aggregate on label (' +str(m) +')...')
            self.final_aggregate(m, begin)
