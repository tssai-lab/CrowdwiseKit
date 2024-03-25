import numpy as np
import random
import re
import math
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn import metrics
from scipy.special import expit

import MACLU


class utils:
    """This class implements several auxiliary methods"""

    @staticmethod
    def convert_arff_to_standard(arff_path, gold_path, attr_path, size_attr, size_label):
        """convert arff file to formatted .gold file and .attr file"""
        arff_file = open(arff_path)
        gold_file = open(gold_path, 'w')
        attr_file = open(attr_path, 'w')
        instance_id = 1
        begin_instance = False
        for line in arff_file:
            strs = re.split(r',|\s', line.strip())
            if begin_instance is True:
                if (strs is not None) and (len(strs) != 0):
                    for i in range(1, size_label + 1):
                        gold_file.write(str(instance_id) + '\t' + str(i) + '\t' + str(int(strs[size_attr + i - 1]) + 1) + '\n')
                    for j in range(0, size_attr):
                        if j == 0:
                            attr_file.write(str(instance_id))
                            attr_file.write(',' + strs[j])
                        elif j == size_attr - 1:
                            attr_file.write(',' + strs[j])
                            attr_file.write('\n')
                        else:
                            attr_file.write(',' + strs[j])
                    instance_id += 1
            else:
                if (strs is not None) and (len(strs) != 0):
                    if (strs[0] == '@data') or (strs[0] == '@DATA'):
                        begin_instance = True

    @staticmethod
    def sigmoid(x):
        sig = expit(x)
        sig = np.minimum(sig, 0.99999999)  # Set upper bound
        sig = np.maximum(sig, 0.00000001)  # Set lower bound
        return sig


class activelearnerMACLU:

    LIKELIHOOD_DIFF_RELATIVE = 0.001
    LEARNRATE_CLASSIFIER = 0.001   # learning rate alpha of theta0 and theta1
    LEARNRATE_WORKER = 0.01
    ERROR_DIFF_CLASSIFIER = 0.0001  # error rate difference of theta0 and theta1
    ERROR_DIFF_WORKER = 0.0001
    EPISILON = 0.5
    MAXROUND = 50

    QUERYNUM = 5000  # limited the number of queries

    def __init__(self, resp_file_path, attr_file_path, instance_num, worker_num, label_num, gold_file_path=None):
        self.resp_file_path = resp_file_path
        self.attr_file_path = attr_file_path
        self.gold_file_path = gold_file_path
        self.instance_num = instance_num
        self.worker_num = worker_num
        self.label_num = label_num
        self.has_gold_file = False
        self.classifier = None
        self.episilon = self.EPISILON
        self.converge_rate = self.LIKELIHOOD_DIFF_RELATIVE
        self.learning_rate_theta0 = self.LEARNRATE_CLASSIFIER
        self.learning_rate_theta1 = self.LEARNRATE_WORKER
        self.error_rate_theta0 = self.ERROR_DIFF_CLASSIFIER
        self.error_rate_theta1 = self.ERROR_DIFF_WORKER
        self.maxround = self.MAXROUND
        self.query_num = self.QUERYNUM
        self.theta0 = dict()  # weights for each classifier on each label
        self.theta1 = dict()  # weights for each worker on each label

        self.gold_matrix = None
        # self.worker_matrix = None
        self.instance_matrix = None
        self.label_record = None  # indication of annotated labels
        self.aggregated_matrix = None
        self.predicted_matrix = None
        self.result_matrix = None

        self.labeled_X = dict()  # the features of labeled instances for each label
        self.labeled_X_fulllabeled = np.array([])  # used to create enhanced feature of each instance
        self.labeled_Y = dict()  # the aggregated labels for each labeled instance
        self.labeled_enhancedX = dict()

        self.labeled_X_worker = dict()
        self.labeled_Y_worker = dict()

        self.attr_map = dict()
        self.X = None  # features of all instances (contains labeled and unlabeled), converted from attr_map
        self.enhancedX = None

        self.indicesNextInstance = None
        self.indicesNextLabel = None
        self.indicesNextWoker = None

        self.attr_num = 0

        self.nns = None
        self.poolExtraData = None

        self.utils = utils()

    def initialize(self):
        print('The dataset contains %d crowd workers, %d instances and %d labels' % (
            self.worker_num, self.instance_num, self.label_num))

        for i in range(1, self.label_num + 1):
            self.theta0[i] = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
            for j in range(1, self.worker_num + 1):
                self.theta1['w%d,%d' % (j, i)] = LogisticRegression(C=1, max_iter=1000, penalty='l2', solver='liblinear', class_weight='balanced')

        self.gold_matrix = np.zeros((self.instance_num, self.label_num))
        # self.worker_matrix = np.zeros((self.worker_num, self.instance_num, self.label_num))
        self.instance_matrix = np.zeros((self.instance_num, self.worker_num, self.label_num))
        self.label_record = np.zeros((self.instance_num, self.label_num, self.worker_num))  # indication of annotated labels
        self.aggregated_matrix = np.zeros((self.instance_num, self.label_num))
        self.predicted_matrix = np.zeros((self.instance_num, self.label_num))
        self.result_matrix = np.zeros((self.instance_num, self.label_num))

        '''initialize the gold matrix of instances'''
        if self.gold_file_path is not None:
            print('There is a gold file!')
            gold_f = open(self.gold_file_path)
            for line in gold_f:
                strs = line.split()
                if len(strs) <= 2:
                    gold_f.close()
                    print('Error formatted gold file', end='\n')
                    return None
                self.gold_matrix[int(strs[0])-1][int(strs[1])-1] = int(strs[2])
            self.has_gold_file = True
            gold_f.close()

        '''initialize feature map'''
        attr_f = open(self.attr_file_path)
        for line in attr_f:
            strs = re.split(r',| ', line.strip())
            tmp_fea = []
            for s in strs[1:]:
                tmp_fea.append(float(s))
            self.attr_map[int(strs[0])] = tmp_fea
        self.attr_num = len(self.attr_map[1])
        attr_f.close()

        self.X = np.zeros((1, self.attr_num))
        for i in range(1, self.instance_num + 1):
            self.X = np.vstack((self.X, self.attr_map[i]))
        self.X = np.delete(self.X, 0, axis=0)

        resp_f = open(self.resp_file_path)
        for line in resp_f:
            strs = line.split()
            if len(strs) <= 3:
                resp_f.close()
                print('Error formatted response file', end='\n')
                return None
            # self.worker_matrix[int(strs[0])-1][int(strs[1])-1][int(strs[2])-1] = int(strs[3])
            self.instance_matrix[int(strs[1])-1][int(strs[0])-1][int(strs[2])-1] = int(strs[3])
            if int(strs[3]) != 0:
                self.label_record[int(strs[1])-1][int(strs[2])-1][int(strs[0])-1] = 1
        resp_f.close()

        '''initialize aggregated matrix'''
        for l in range(1, self.label_num+1):
            for i in range(1, self.instance_num+1):
                if sum(self.label_record[i-1][l-1])>0:
                    '''initialize aggregated labels using MV'''
                    neg = np.sum(self.instance_matrix[i-1, :, l-1] == 1)
                    pos = np.sum(self.instance_matrix[i-1, :, l-1] == 2)
                    if neg > pos:
                        tmp_Y2 = 1
                    elif neg < pos:
                        tmp_Y2 = 2
                    else:
                        tmp_Y2 = random.choice([1, 2])
                    self.aggregated_matrix[i-1][l-1] = tmp_Y2
                    self.result_matrix[i-1][l-1] = self.aggregated_matrix[i-1][l-1]

        self.update_enhancedX()

        '''choose labeled data to generate initial labeled_X (and labeled_enhancedX) and labeled_Y'''
        for l in range(1, self.label_num + 1):
            is_first_line = True
            tmp_enhancedX = np.array([])
            tmp_Y = np.array([])
            for i in range(1, self.instance_num+1):
                if sum(self.label_record[i-1][l-1]) > 0:
                    if is_first_line:
                        tmp_enhancedX = np.array(self.enhancedX[i-1])
                        tmp_Y = np.array(self.aggregated_matrix[i-1][l-1])
                        is_first_line = False
                    else:
                        tmp_enhancedX = np.row_stack((tmp_enhancedX, np.array(self.enhancedX[i-1])))
                        tmp_Y = np.row_stack((tmp_Y, np.array(self.aggregated_matrix[i-1][l-1])))
            self.labeled_Y[l] = tmp_Y
            self.labeled_enhancedX[l] = tmp_enhancedX
            self.theta0[l].fit(self.labeled_enhancedX[l], self.labeled_Y[l].ravel())

        for w in range(1, self.worker_num + 1):
            for l in range(1, self.label_num + 1):
                is_first_line = True
                tmp_X = np.array([])
                tmp_Y = np.array([])
                for i in range(1, self.instance_num + 1):
                    if self.label_record[i-1][l-1][w-1] == 1:
                        if is_first_line:
                            tmp_X = np.array(self.attr_map[i])
                            if self.instance_matrix[i-1][w-1][l-1] == self.aggregated_matrix[i-1][l-1]:
                                tmp_Y = np.array([2])
                            else:
                                tmp_Y = np.array([1])
                            is_first_line = False
                        else:
                            tmp_X = np.row_stack((tmp_X, np.array(self.attr_map[i])))
                            if self.instance_matrix[i-1][w-1][l-1] == self.aggregated_matrix[i-1][l-1]:
                                tmp_Y = np.row_stack((tmp_Y, np.array([2])))
                            else:
                                tmp_Y = np.row_stack((tmp_Y, np.array([1])))
                self.labeled_X_worker['w%d,%d' % (w, l)] = tmp_X
                self.labeled_Y_worker['w%d,%d' % (w, l)] = tmp_Y
                self.theta1['w%d,%d' % (w, l)].fit(self.labeled_X_worker['w%d,%d' % (w, l)], self.labeled_Y_worker['w%d,%d' % (w, l)].ravel())

    def update_enhancedX(self):
        """initialize the enhanced features for all the instances based on the labeled data"""
        tmp_X_fulllabeled = np.zeros((1, self.attr_num))  # features of the instances which have all labels annotated
        for i in range(self.instance_num):
            anno_label_num = 0 # count of labels annotated
            for l in range(self.label_num):
                if sum(self.label_record[i][l])>0:
                    anno_label_num += 1
            if anno_label_num == self.label_num:
                tmp_X_fulllabeled = np.row_stack((tmp_X_fulllabeled, np.array(self.attr_map[i+1])))
        self.labeled_X_fulllabeled = np.delete(tmp_X_fulllabeled, 0, axis=0)

        self.nns = NearestNeighbors(n_neighbors=11, metric='euclidean')
        # calculate enhanced codes for dataset
        self.poolExtraData = np.zeros((1, self.label_num))
        self.nns.fit(self.labeled_X_fulllabeled)
        distances, indices = self.nns.kneighbors(self.X)
        phi = np.zeros((1, self.label_num))
        for i in range(len(distances)):
            sum_dis = 0
            for j in range(1,6):
                sum_dis += distances[i][j]
            weight = distances[i][1:6] / sum_dis
            weight = np.flipud(weight)
            tmp_phi = np.zeros((1, self.label_num))
            for j in range(1, 6):
                tmp_phi += weight[j-1] * self.aggregated_matrix[indices[i][j]]
            phi = np.row_stack((phi, tmp_phi))
        phi = np.delete(phi, 0, axis=0)
        self.enhancedX = np.hstack([self.X, phi])

    def e_step(self, l):
        for i in range(self.instance_num):
            if np.sum(self.label_record[i][l-1]) > 0:
                pr_neg = self.theta0[l].predict_proba(self.enhancedX[i].reshape(1, -1))[0][0]
                pr_pos = self.theta0[l].predict_proba(self.enhancedX[i].reshape(1, -1))[0][1]
                for w in range(self.worker_num):
                    if self.instance_matrix[i][w][l-1] == 2:
                        pr_pos = pr_pos * self.theta1['w%d,%d' % (w+1, l)].predict_proba(self.X[i].reshape(1, -1))[0][1]
                        pr_neg = pr_neg * self.theta1['w%d,%d' % (w+1, l)].predict_proba(self.X[i].reshape(1, -1))[0][0]
                    elif self.instance_matrix[i][w][l-1] == 1:
                        pr_pos = pr_pos * self.theta1['w%d,%d' % (w+1, l)].predict_proba(self.X[i].reshape(1, -1))[0][0]
                        pr_neg = pr_neg * self.theta1['w%d,%d' % (w+1, l)].predict_proba(self.X[i].reshape(1, -1))[0][1]
                if pr_pos >= pr_neg:
                    self.aggregated_matrix[i][l-1] = 2
                else:
                    self.aggregated_matrix[i][l-1] = 1
                self.result_matrix[i][l-1] = self.aggregated_matrix[i][l-1]

        #  update labeled_enhancedX, labeled_Y, labeled_X_worker, labeled_Y_worker
        self.update_trainset()

    def update_trainset(self):
        for l in range(1, self.label_num + 1):
            is_first_line = True
            tmp_enhancedX = np.array([])
            tmp_Y = np.array([])
            for i in range(1, self.instance_num+1):
                if sum(self.label_record[i-1][l-1]) > 0:
                    if is_first_line:
                        tmp_enhancedX = np.array(self.enhancedX[i-1])
                        tmp_Y = np.array(self.aggregated_matrix[i-1][l-1])
                        is_first_line = False
                    else:
                        tmp_enhancedX = np.row_stack((tmp_enhancedX, np.array(self.enhancedX[i-1])))
                        tmp_Y = np.row_stack((tmp_Y, np.array(self.aggregated_matrix[i-1][l-1])))
            self.labeled_Y[l] = tmp_Y
            self.labeled_enhancedX[l] = tmp_enhancedX

        for w in range(1, self.worker_num + 1):
            for l in range(1, self.label_num + 1):
                is_first_line = True
                tmp_X = np.array([])
                tmp_Y = np.array([])
                for i in range(1, self.instance_num + 1):
                    if self.label_record[i-1][l-1][w-1] == 1:
                        if is_first_line:
                            tmp_X = np.array(self.attr_map[i])
                            if self.instance_matrix[i-1][w-1][l-1] == self.aggregated_matrix[i-1][l-1]:
                                tmp_Y = np.array([2])
                            else:
                                tmp_Y = np.array([1])
                            is_first_line = False
                        else:
                            tmp_X = np.row_stack((tmp_X, np.array(self.attr_map[i])))
                            if self.instance_matrix[i-1][w-1][l-1] == self.aggregated_matrix[i-1][l-1]:
                                tmp_Y = np.row_stack((tmp_Y, np.array([2])))
                            else:
                                tmp_Y = np.row_stack((tmp_Y, np.array([1])))
                self.labeled_X_worker['w%d,%d' % (w, l)] = tmp_X
                self.labeled_Y_worker['w%d,%d' % (w, l)] = tmp_Y

    def m_step(self, l):

        for w in range(1, self.worker_num + 1):
            self.theta1['w%d,%d' % (w, l)].fit(self.labeled_X_worker['w%d,%d' % (w, l)], self.labeled_Y_worker['w%d,%d' % (w, l)].ravel())
        self.theta0[l].fit(self.labeled_enhancedX[l], self.labeled_Y[l].ravel())
        # diff = 0
        # max_itor = 5000  # max number of iteration
        # error1 = 0  # new error rate
        # error0 = 0  # old error rate
        # m = self.instance_num
        # cnt = 0  # number of iteration
        #
        # '''update theta0'''
        # while cnt <= max_itor:
        #     cnt += 1
        #     for i in range(m):
        #         if np.sum(self.label_record[i][l-1]) > 0:
        #             # calculate difference
        #             diff = self.utils.sigmoid(self.utils.function_h(self.theta0[l], self.enhancedX[i])) - self.aggregated_matrix[i][l-1]
        #
        #             # calculate gradient
        #             enhancedX = np.append(self.enhancedX[i], 1)
        #             # self.theta0[l] -= alpha * diff * enhancedX
        #             for t in range(len(self.theta0[l])):
        #                 self.theta0[l][t] -= self.learning_rate_theta0 * diff * enhancedX[t]
        #
        #     # calculate loss
        #     error1 = 0
        #     for j in range(m):
        #         if np.sum(self.label_record[j][l-1]) > 0:
        #             error1 += (self.aggregated_matrix[j][l-1]-self.utils.sigmoid(self.utils.function_h(self.theta0[l], self.enhancedX[j])))**2/2
        #     # print('error1=', error1)
        #
        #     if abs(error1-error0) < self.error_rate_theta0:
        #         # print('迭代结束，分类器参数迭代次数为',cnt)
        #         break
        #     else:
        #         error0 = error1
        #
        # n = self.worker_num
        #
        # # update theta1
        # for w in range(self.worker_num):
        #     diff1 = 0
        #     error3 = 0  # new error rate
        #     error2 = 0  # old error rate
        #     cnt1 = 0  # number of iteration
        #
        #     while cnt1 <= max_itor:
        #         cnt1 += 1
        #         for i in range(m):
        #             # calculate difference
        #             if self.label_record[i][l-1][w] > 0:
        #                 diff1 = (self.utils.sigmoid(self.utils.function_h(self.theta1['w%d,%d' % (w + 1, l)], self.X[i])*self.instance_matrix[i][w][l-1]) - self.aggregated_matrix[i][l-1]) * self.instance_matrix[i][w][l-1]
        #
        #                 # calculate gradient
        #                 X = np.append(self.X[i], 1)
        #                 for t in range(len(self.theta1['w%d,%d' % (w + 1, l)])):
        #                     self.theta1['w%d,%d' % (w + 1, l)][t] -= self.learning_rate_theta1 * diff1 * X[t]
        #
        #         # calculate loss
        #         error3 = 0
        #         for j in range(m):
        #             if self.label_record[j][l - 1][w] > 0:
        #                 error3 += (self.instance_matrix[j][w][l-1] - self.utils.sigmoid(
        #                     self.utils.function_h(self.theta1['w%d,%d' % (w + 1, l)], self.X[j]))) ** 2 / 2
        #         # print('error3=', error3)
        #
        #         if abs(error3-error2) < self.error_rate_theta1:
        #             # print('工人',w+1,'的参数迭代次数为',cnt1)
        #             break
        #         else:
        #             error2 = error3

    def loglikelihood(self, l):
        log_like = 0
        for i in range(self.instance_num):
            if np.sum(self.label_record[i][l-1]) > 0:
                p = []
                for k in range(1, 3):
                    prod = 1.0
                    for w in range(self.worker_num):
                        if self.instance_matrix[i][w][l - 1] == 0:
                            continue
                        elif self.instance_matrix[i][w][l - 1] == k:
                            prod *= self.theta1['w%d,%d' % (w + 1, l)].predict_proba(self.X[i].reshape(1, -1))[0][1]
                        else:
                            prod *= self.theta1['w%d,%d' % (w + 1, l)].predict_proba(self.X[i].reshape(1, -1))[0][0]
                    prior_k = self.theta0[l].predict_proba(self.enhancedX[i].reshape(1, -1))[0][k-1]
                    p.append(prod * prior_k)
                tmp_likelihood = sum(p)
                log_like += math.log(tmp_likelihood)
        return log_like

    def em(self, l):
        cnt = 1
        last_likelihood = -999999999
        curr_likehihood = self.loglikelihood(l)
        # print('MACLU on label (' + str(l) + ') initial log-likelihood = ' + str(curr_likehihood))
        while (cnt <= self.maxround) and (abs(curr_likehihood - last_likelihood) / abs(last_likelihood) > self.converge_rate):
            self.e_step(l)
            self.m_step(l)
            last_likelihood = curr_likehihood
            curr_likehihood = self.loglikelihood(l)
            # print('MACLU on label (' + str(l) + ') round (' + str(cnt) + ') log-likelihood = ' + str(curr_likehihood))
            cnt += 1

    def select_next_instance(self):
        """select next example"""
        LabelCard = np.sum(self.aggregated_matrix == 2)
        LabeledData = 0
        for i in range(self.instance_num):
            if np.sum(self.instance_matrix[i]) > 0:
                LabeledData += 1
        aveLabelCard = LabelCard / LabeledData

        Sx = []
        ins_fulllabeled = 0
        for i in range(self.instance_num):
            if np.sum(self.label_record[i]) < (self.worker_num * self.label_num):
                LCI = abs(np.sum(self.result_matrix[i] == 2) - aveLabelCard)
                anno = max(self.episilon, np.sum(self.label_record[i] > 0))
                CI = LCI / anno
                Ux = 0
                for l in range(self.label_num):
                    Ux += abs(self.theta0[l+1].predict_proba(self.enhancedX[i].reshape(1, -1))[0][1] - 0.5)
                Ux = Ux / self.label_num
                tmp_sx = CI/Ux
            else:
                tmp_sx = float('-inf')
                ins_fulllabeled += 1
            Sx.append(tmp_sx)
        if ins_fulllabeled == self.instance_num:
            print('Error: There is no instance to select since all the instances are already annotated!',end='\n')
            return None
        indices = np.where(Sx == max(Sx))
        self.indicesNextInstance = random.choice(indices[0])+1
        # print('The selected instance is', self.indicesNextInstance)
        return self.indicesNextInstance

    def select_next_label(self):
        """select next label"""
        select_l = []
        for l in range(self.label_num):
            if np.sum(self.label_record[self.indicesNextInstance-1][l]) < self.worker_num:
                tmp_l = 0.5 - abs(self.theta0[l+1].predict_proba(self.enhancedX[self.indicesNextInstance-1].reshape(1, -1))[0][1] - 0.5)
            else:
                tmp_l = float('-inf')
            select_l.append(tmp_l)
        indices = np.where(select_l == max(select_l))
        self.indicesNextLabel = random.choice(indices[0])+1
        # print('The selected label is', self.indicesNextLabel)
        return self.indicesNextLabel

    def select_next_worker(self):
        """select next worker"""
        expertise = []
        for k in range(self.worker_num):
            if np.sum(self.label_record[self.indicesNextInstance-1][self.indicesNextLabel-1][k]) == 0:
                tmp_w = self.theta1['w%d,%d' % (k+1, self.indicesNextLabel)].predict_proba(self.X[self.indicesNextInstance-1].reshape(1, -1))[0][1]
            else:
                tmp_w = float('-inf')
            expertise.append(tmp_w)
        indices = np.where(expertise == max(expertise))
        self.indicesNextWoker = random.choice(indices[0])+1
        # print('The selected worker is', self.indicesNextWoker)
        return self.indicesNextWoker

    def select_next(self):
        next_instance = self.select_next_instance()
        next_label = self.select_next_label()
        next_worker = self.select_next_worker()
        return next_instance, next_label, next_worker

    def update(self, anno):
        #  update instance_matrix, label_record
        if self.instance_matrix[self.indicesNextInstance-1][self.indicesNextWoker-1][self.indicesNextLabel-1] != 0 or self.label_record[self.indicesNextInstance-1][self.indicesNextLabel-1][self.indicesNextWoker-1] != 0:
            print('ERROR: The label ', self.indicesNextLabel, ' of instance ', self.indicesNextInstance, ' provided by worker ', self.indicesNextWoker, ' has been already obtained!')
            return None
        self.instance_matrix[self.indicesNextInstance-1][self.indicesNextWoker-1][self.indicesNextLabel-1] = anno
        self.label_record[self.indicesNextInstance-1][self.indicesNextLabel-1][self.indicesNextWoker-1] = 1
        self.predicted_matrix[self.indicesNextInstance-1][self.indicesNextLabel-1] = 0

    def infer(self):
        #  infer the aggregated labels for labeled instances
        for i in range(1, self.label_num+1):
            self.em(i)

        #  infer the predicted labels for unlabeled instances
        for i in range(self.instance_num):
            for l in range(self.label_num):
                if np.sum(self.label_record[i][l]) == 0:
                    p = self.theta0[l+1].predict_proba(self.enhancedX[i].reshape(1, -1))[0][1]
                    if p < 0.5:
                        self.predicted_matrix[i][l] = 1
                    else:
                        self.predicted_matrix[i][l] = 2
                    self.result_matrix[i][l] = self.predicted_matrix[i][l]

    def print_aggregate_accuracy(self):
        # acc = metrics.accuracy_score(self.gold_matrix, self.result_matrix)
        acc = 0
        to = 0
        co = 0
        for i in range(self.instance_num):
            for l in range(self.label_num):
                if self.aggregated_matrix[i][l] > 0:
                    to += 1
                    if self.aggregated_matrix[i][l] == self.gold_matrix[i][l]:
                        co += 1
        # print('total number = ', to)
        # print('correct number = ', co)
        acc = co / to
        return acc

    def print_predict_accuracy(self):
        acc = 0
        to = 0
        co = 0
        for i in range(self.instance_num):
            for l in range(self.label_num):
                if self.predicted_matrix[i][l] > 0:
                    to += 1
                    if self.predicted_matrix[i][l] == self.gold_matrix[i][l]:
                        co += 1
        # print('total number = ', to)
        # print('correct number = ', co)
        acc = co / to
        return acc

    def print_total_accuracy(self):
        acc = metrics.accuracy_score(self.gold_matrix, self.result_matrix)
        return acc

    def get_aggregated_labels(self):
        return self.aggregated_matrix

    def get_predicted_labels(self):
        return self.predicted_matrix

    def get_result_labels(self):
        return self.result_matrix

    def set_converge_rate(self, val):
        self.converge_rate = val

    def set_learning_rate_classifier(self, val):
        self.learning_rate_theta0 = val

    def set_learning_rate_worker(self, val):
        self.learning_rate_theta1 = val

    def set_error_rate_classifier(self, val):
        self.error_rate_theta0 = val

    def set_error_rate_worker(self, val):
        self.error_rate_theta1 = val

    def set_epsilon(self, val):
        self.episilon = val

    def set_maxround(self, val):
        self.maxround = val

    def set_querynum(self, val):
        self.query_num = val


if __name__ == '__main__':
    print('test begin...')
