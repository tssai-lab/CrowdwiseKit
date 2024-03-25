# -*- coding: utf-8 -*-
import numpy
import sklearn.decomposition

class Label:
    """
    The base class of Label
    """
    INVALID_ID = 0
    INVALID_VAL = -1
    SINGLE_LABLE = 3 # because there are 3 items in a line in a response file
    MULTI_LABEL = 4

    def __init__(self, id):
        self.id = id
        self.val = 0
        self.inst_id = 0
        self.worker_id = 0


class Worker:
    """
    The base class of crowdsourced workers
    """
    INVALID_ID = 0
    GOLD = 9999999 # gold worker that provides the ground truth
    AGGR = 9999998 # aggregated worker

    def __init__(self, id):
        self.id = id
        self.label_dict = dict()
        self.sorted_label_dicts = {1:dict()}

    def get_label(self, label_info):
        if label_info in self.label_dict:
            return self.label_dict[label_info]
        return None

    def add_label(self, label):
        tpl = (label.id, label.inst_id, label.worker_id)
        if tpl not in self.label_dict:
            self.label_dict.setdefault(tpl, label)
        target_dict = self.sorted_label_dicts.get(label.id)
        if target_dict == None:
            target_dict = dict()
            self.sorted_label_dicts.setdefault(label.id, target_dict)
        target_dict.setdefault(tpl, label)

    def get_sorted_label_dict_size(self):
        return len(self.sorted_label_dicts)

    def get_label_list(self):
        list = []
        for (k, v) in self.label_dict.items():
            list.append(v)
        return list

    def get_label_val_for_inst(self, inst_id, label_id):
        tpl = (label_id, inst_id, self.id)
        label = self.get_label(tpl)
        if label == None:
            return 0
        else:
            return label.val

class Instance:
    """
    The base class of instance
    """
    INVALID_ID = 0

    def __init__(self, id):
        self.id = id
        self.true_label_dict = dict()
        self.intg_label_dict = dict()
        self.sorted_label_dicts = {1:list()}

    def get_true_label(self, label_id):
        return self.true_label_dict.get(label_id)

    def add_true_label(self, label):
        if label.id not in self.true_label_dict:
            self.true_label_dict.setdefault(label.id, label)

    def get_true_label_set(self):
        return self.true_label_dict

    def add_noisy_label(self, label):
        if self.sorted_label_dicts.get(label.id) == None:
            self.sorted_label_dicts.setdefault(label.id, list())
            self.sorted_label_dicts[label.id].append(label)
            return
        for l in self.sorted_label_dicts[label.id]:
            if (l.id == label.id) and (l.worker_id == label.worker_id) and (l.inst_id == label.inst_id)\
                and (l.val == label.val):
                return
        self.sorted_label_dicts[label.id].append(label)

    def get_noisy_labels(self, label_id):
        return  self.sorted_label_dicts[label_id]

    def get_noisy_label(self, label_id, worker_id):
        labels = self.sorted_label_dicts[label_id]
        for l in labels:
            if (l.worker_id == worker_id):
                return l

    def add_integrated_label(self, label):
        if label.id in self.intg_label_dict:
            self.intg_label_dict[label.id] = label
        else:
            self.intg_label_dict.setdefault(label.id, label)

    def get_integrated_label(self, label_id):
        if label_id in self.intg_label_dict:
            return self.intg_label_dict[label_id]
        return None

    def get_label_id_list(self):
        id_list =[]
        for (k, v) in self.sorted_label_dicts.items():
            id_list.append(k)
        return id_list

    def equal_integrated_true(self, label_id):
        if self.true_label_dict[label_id].val == self.intg_label_dict[label_id].val:
            return 1
        return 0

    def all_match_integrated_true(self):
        id_list = self.get_label_id_list()
        for label_id in id_list:
            if self.equal_integrated_true(label_id) == 0:
                return 0
        return 1

    def get_worker_id_set(self):
        s = set()
        for (k, v) in self.sorted_label_dicts.items():
            for l in v:
                s.add(l.worker_id)
        return s

class Dataset:
    """
    The base class of a data set
    """
    def __init__(self):
        # info for label creation from files
        self._label_next_id_val = {1: 1}  # the first term is label id, the second one is its value
        self._label_id_name_dict = dict()  # mapping label_id and label name
        self._label_val_name_dicts = {1: dict()}  # the first term is label id, the second one is a dictionary to map values
        # info for worker creation from files
        self._worker_next_id = 1  # the worker id starts from 1
        self._worker_id_name_dict = dict()  # mapping worker_id and worker name
        # info for instance creation from files
        self._instance_next_id = 1  # the instance id starts from 1
        self._instance_id_name_dict = dict()
        self.create_from_files = False
        # for general usage
        self.inst_dict = dict()
        self.worker_dict = dict()
        self.label_info_dict = {1:set()}

    # functions for create label from files
    def fetch_label_val_id_by_name(self, label_id, val_name):
        if self._label_val_name_dicts.get(label_id) == None:
            self._label_val_name_dicts.setdefault(label_id, dict())
        val_dict = self._label_val_name_dicts[label_id]
        for (k, v) in val_dict.items():
            if v == val_name:
                return k
        val_id = self._label_next_id_val[label_id]
        self._label_val_name_dicts[label_id].setdefault(val_id, val_name)
        self._label_next_id_val[label_id] += 1
        return val_id

    def fetch_label_id_by_name(self, label_name):
        max_label_id = 0
        for (k, v) in self._label_id_name_dict.items():
            if k > max_label_id:
                max_label_id = k
            if v == label_name:
                return k
        max_label_id += 1
        if self._label_next_id_val.get(max_label_id) == None:
            self._label_next_id_val.setdefault(max_label_id, 1)
        self._label_id_name_dict.setdefault(max_label_id, label_name)
        return max_label_id

    def get_label_id_by_name(self, label_name):
        for (k, v) in self._label_id_name_dict.items():
            if v == label_name:
                return k
        return Label.INVALID_ID

    def get_label_name_by_id(self, label_id):
        return self._label_id_name_dict[label_id]

    def get_label_val_name_by_id(self, label_id, val_id):
        return self._label_val_name_dicts[label_id][val_id]

    def label_info_confirm(self):
        # add label info
        for (k, v) in self._label_val_name_dicts.items():
            for (k2, v2) in v.items():
                self.add_label_info(k, k2)

    # functions for create workers from files
    def fetch_worker_id_by_name(self, name):
        for (k, v) in self._worker_id_name_dict.items():
            if v == name:
                return k
        id = self._worker_next_id
        self._worker_id_name_dict.setdefault(id, name)
        self._worker_next_id += 1
        return id

    def get_worker_name_by_id(self, id):
        return self._worker_id_name_dict[id]

    # functions for create instances from files
    def fetch_instance_id_by_name(self, name):
        for (k, v) in self._instance_id_name_dict.items():
            if v == name:
                return k
        id = self._instance_next_id
        self._instance_id_name_dict.setdefault(id, name)
        self._instance_next_id += 1
        return id

    def get_instance_id_by_name(self, name):
        for (k, v) in self._instance_id_name_dict.items():
            if v == name:
                return k
        return Instance.INVALID_ID

    def get_instance_name_by_id(self, id):
        return self._instance_id_name_dict[id]

    # general usage functions
    def get_instance(self, id):
        return self.inst_dict.get(id)

    def get_worker(self, id):
        return self.worker_dict.get(id)

    def add_instance(self, inst):
        if self.get_instance(inst.id) == None:
            self.inst_dict.setdefault(inst.id, inst)

    def add_worker(self, worker):
         if self.get_worker(worker.id) == None:
             self.worker_dict.setdefault(worker.id, worker)

    def get_instance_size(self):
        return len(self.inst_dict)

    def get_worker_size(self):
        return len(self.worker_dict)

    def is_multi_class(self):
        for (k, v) in self.worker_dict.items():
            if v.get_sorted_label_dict_size() > 1:
                return True
        return False

    def add_label_info(self, label_id, val):
        if self.label_info_dict.get(label_id) == None:
            self.label_info_dict.setdefault(label_id, set())
        self.label_info_dict[label_id].add(val)

    def get_label_id_size(self):
        return len(self.label_info_dict)

    def get_label_val_size(self, label_id):
        return len(self.label_info_dict[label_id])

    def get_num_explained_components_by_PCA(self, rho):
        row_num = 0
        for (k, v) in self.inst_dict.items():
            s = v.get_worker_id_set()
            row_num += len(s)
        num_labels = self.get_label_id_size()
        correlation = numpy.ndarray(shape=(row_num, num_labels), dtype=float, order='C')
        correlation.fill(0)
        r = 0
        for (k, v) in self.inst_dict.items():
            s = v.get_worker_id_set()
            for worker_id in s:
                for label_id in range(1, num_labels+ 1):
                    label = v.get_noisy_label(label_id, worker_id)
                    correlation[r][label_id-1] = label.val
                r += 1
        pca = sklearn.decomposition.PCA(n_components=num_labels, svd_solver='full')
        pcaresult = pca.fit(correlation)
        num_components = 0
        explain = 0.0
        for pos in range (0, num_labels):
            explain += pcaresult.explained_variance_ratio_[pos]
            if (explain - rho) >= -0.05: # explain is around rho, the condition also satisfies
                num_components = pos+1
                break
        return num_components
