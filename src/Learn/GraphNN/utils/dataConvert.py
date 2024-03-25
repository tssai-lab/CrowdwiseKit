# arff格式统一转换为.gold,.resp,.fea格式
# 2022.12.13修改, .fea文件最后一列增加样本真实标签

import re
import pickle
import numpy as np
from workerCreator import Worker, Workers


class syntheticdataProcessor:
    def __init__(self, arff_path, gold_path, feature_path, size_attrib, size_label, instance_id = 1):
        self.arff_path = arff_path  # arff文件地址
        self.gold_path = gold_path  # 要保存的.gold文件地址
        self.feature_path = feature_path  # 要保存的.fea文件地址
        self.size_attrib = size_attrib  # 特征数
        self.size_label = size_label  # 标签数
        self.instance_id = instance_id  # 样本起始编号，默认为1

    # -*- coding: utf-8 -*-
    # To divide the arff file into two files: 'gold' file and 'fea' file
    # 'gold' file: to store the golden labels of instances, formatted as 'instance_id label_id'
    # 'fea' file: to store the features of instances, formatted as 'instance_id,feature_1,feature_2,..., ground_truth'
    # input: path of arff file, gold file and fea file
    # @author: Ming Wu (wuming@njust.edu.cn)
    def arff_to_gold_and_fea(self):
        arff_file = open(self.arff_path)
        gold_file = open(self.gold_path, 'w')
        fea_file = open(self.feature_path, 'w')
        begin_instance = False
        for line in arff_file:
            strs = re.split(r',|\s', line.strip())
            if begin_instance is True:
                if (strs is not None) and (len(strs) != 0):
                    if len(strs) == (self.size_attrib + self.size_label):
                        for i in range(1, self.size_label + 1):
                            gold_file.write(
                                str(self.instance_id) + '\t' + str(int(strs[self.size_attrib + i - 1])) + '\n')
                        for j in range(0, self.size_attrib):
                            if j == 0:
                                fea_file.write(str(self.instance_id))
                                fea_file.write(',' + strs[j])
                            elif j == self.size_attrib - 1:
                                fea_file.write(',' + strs[j] + ',' + strs[j + 1])
                                fea_file.write('\n')
                            else:
                                fea_file.write(',' + strs[j])
                        self.instance_id += 1
            else:
                if (strs is not None) and (len(strs) != 0):
                    if (strs[0] == '@data') or (strs[0] == '@DATA'):
                        begin_instance = True
        fea_file.close()
        gold_file.close()
        arff_file.close()

    # 根据众包工作者信息生成.resp文件 （二分类）
    def arff_to_resp(self, resp_path, worker_path):
        f = open(worker_path, 'rb')
        wks = pickle.load(f)
        f.close()
        g = open(self.gold_path)
        r = open(resp_path, 'w')
        cnt = 0
        for line in g:
            strs = line.split()
            label = 1
            gt = int(strs[1])
            for wk in wks.workers:
                rand = np.random.uniform(0,1)
                if rand <= wk.cms[label-1][gt][gt]:
                    ans = gt
                else:
                    ans = 1 - gt
                r.write(str(wk.workerID) + '\t' + strs[0] + '\t' + str(ans) + '\n')
        g.close()
        r.close()

    # 根据众包工作者信息生成.resp文件 （多分类）
    def arff_to_resp_multi(self, resp_path, worker_path, class_num):
        f = open(worker_path, 'rb')
        wks = pickle.load(f)
        f.close()
        g = open(self.gold_path)
        r = open(resp_path, 'w')
        cnt = 0
        for line in g:
            strs = line.split()
            label = 1
            gt = int(strs[1])
            for wk in wks.workers:
                rand = np.random.uniform(0,1)
                if rand <= wk.cm[gt][gt]:
                    ans = gt
                else:
                    class_set = [i for i in range(class_num)]
                    class_set.remove(gt)
                    max_prob = wk.cm[gt][gt]
                    for i in class_set:
                        if rand <= (max_prob + wk.cm[gt][i]):
                            ans = i
                            break
                        else:
                            max_prob = max_prob + wk.cm[gt][i]
                            continue
                r.write(str(wk.workerID) + '\t' + strs[0] + '\t' + str(ans) + '\n')
        g.close()
        r.close()


class realworlddataProcessor:
    def __init__(self, arff_path, old_gold_path, gold_path, feature_path, size_attrib, size_label):
        self.arff_path = arff_path  # arff文件地址
        self.old_gold_path = old_gold_path # 原始.gold文件地址
        self.gold_path = gold_path  # 要保存的.gold文件地址
        self.feature_path = feature_path  # 要保存的.fea文件地址
        self.size_attrib = size_attrib  # 特征数
        self.size_label = size_label  # 标签数
        self.instance_id = 1

    def arff_to_gold_and_fea(self):
        old_gold_file = open(self.old_gold_path)
        # 用于生成样本编号对应字典{old_ID: new_ID}
        beginID = 1
        sample_ID_map = dict()
        for line in old_gold_file:
            strs = re.split(r',|\s', line.strip())
            if (strs is not None) and (len(strs) == 2):
                sample_ID_map[beginID] = int(strs[0])
                beginID += 1
        self.sample_ID_map = sample_ID_map
        nsample = beginID - 1
        old_gold_file.close()

        arff_file = open(self.arff_path)
        gold_file = open(self.gold_path, 'w')
        fea_file = open(self.feature_path, 'w')
        begin_instance = False
        for line in arff_file:
            strs = re.split(r',|\s', line.strip())
            if begin_instance is True:
                if (strs is not None) and (len(strs) != 0):
                    if len(strs) == (self.size_attrib + self.size_label):
                        for i in range(1, self.size_label + 1):
                            gold_file.write(
                                str(sample_ID_map[self.instance_id]) + '\t' + str(int(strs[self.size_attrib + i - 1])) + '\n')
                        for j in range(0, self.size_attrib):
                            if j == 0:
                                fea_file.write(str(sample_ID_map[self.instance_id]))
                                fea_file.write(',' + strs[j])
                            elif j == self.size_attrib - 1:
                                fea_file.write(',' + strs[j] + ',' + strs[j + 1])
                                fea_file.write('\n')
                            else:
                                fea_file.write(',' + strs[j])
                        self.instance_id += 1
            else:
                if (strs is not None) and (len(strs) != 0):
                    if (strs[0] == '@data') or (strs[0] == '@DATA'):
                        begin_instance = True
        fea_file.close()
        gold_file.close()
        arff_file.close()


if __name__ == '__main__':
    # arff_path = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/real-world/leaves/alder.arffx'
    # gold_path = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/real-world/leaves/processed/alder.gold'
    # old_gold_path = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/real-world/leaves/alder.gold.txt'
    # feature_path = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/real-world/leaves/processed/alder.fea'

    # arff_path = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/real-world/leaves/eucalyptus.arffx'
    # gold_path = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/real-world/leaves/processed/eucalyptus.gold'
    # old_gold_path = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/real-world/leaves/eucalyptus.gold.txt'
    # feature_path = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/real-world/leaves/processed/eucalyptus.fea'

    # arff_path = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/real-world/leaves/maple.arffx'
    # gold_path = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/real-world/leaves/processed/maple.gold'
    # old_gold_path = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/real-world/leaves/maple.gold.txt'
    # feature_path = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/real-world/leaves/processed/maple.fea'

    # arff_path = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/real-world/leaves/oak.arffx'
    # gold_path = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/real-world/leaves/processed/oak.gold'
    # old_gold_path = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/real-world/leaves/oak.gold.txt'
    # feature_path = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/real-world/leaves/processed/oak.fea'
    #
    # rdp = realworlddataProcessor(arff_path, old_gold_path, gold_path, feature_path, 64, 1)
    # rdp.arff_to_gold_and_fea()

    # arff_path = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/synthetic/biodeg/biodeg.arff'
    # resp_path = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/synthetic/biodeg/processed/biodeg.resp'
    # gold_path = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/synthetic/biodeg/processed/biodeg.gold'
    # feature_path = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/synthetic/biodeg/processed/biodeg.fea'

    # arff_path = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/synthetic/ionosphere/ionosphere.arff'
    # resp_path = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/synthetic/ionosphere/processed/ionosphere.resp'
    # gold_path = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/synthetic/ionosphere/processed/ionosphere.gold'
    # feature_path = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/synthetic/ionosphere/processed/ionosphere.fea'

    # worker_path = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/workers/workers_information_noisy.txt'
    # sdp = syntheticdataProcessor(arff_path, gold_path, feature_path, 41, 1)
    # sdp.arff_to_gold_and_fea()
    # sdp.arff_to_resp(resp_path, worker_path)

    arff_path = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/synthetic/waveform/waveform.arff'
    resp_path = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/synthetic/waveform/processed/waveform.resp'
    gold_path = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/synthetic/waveform/processed/waveform.gold'
    feature_path = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/synthetic/waveform/processed/waveform.fea'

    worker_path = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/workers/workers_information_noisy_waveform.txt'
    sdp = syntheticdataProcessor(arff_path, gold_path, feature_path, 40, 1)
    sdp.arff_to_gold_and_fea()
    sdp.arff_to_resp_multi(resp_path, worker_path, class_num=3)