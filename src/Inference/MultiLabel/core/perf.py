# -*- coding: utf-8 -*-

from Inference.MultiLabel.core import data

class Evaluation:
    """
    Performance evaluation
    """
    def __init__(self, dataset = None):
        self.dataset = dataset

    def set_dataset(self, dataset):
        self.dataset = dataset

    def get_accuracy(self):
        num_instance = self.dataset.get_instance_size()
        acc_per_instance = []
        for inst_id in range(1, num_instance + 1):
            inst = self.dataset.get_instance(inst_id)
            label_ids = inst.get_label_id_list()
            correct = 0.0
            for label_id in label_ids:
                correct += inst.equal_integrated_true(label_id)
            acc_per_instance.append(correct / float(len(label_ids)))
        return sum(acc_per_instance) / float(num_instance)

    def get_accuracy_on_label(self, label_id):
        num_instance = self.dataset.get_instance_size()
        total = 0.0
        correct = 0.0
        for inst_id in range(1, num_instance + 1):
            inst = self.dataset.get_instance(inst_id)
            total += 1.0
            correct += inst.equal_integrated_true(label_id)
        return correct/total

    def get_subset_accuracy(self):
        num_instance = self.dataset.get_instance_size()
        correct = 0.0
        for inst_id in range(1, num_instance + 1):
            inst = self.dataset.get_instance(inst_id)
            correct += inst.all_match_integrated_true()
        return correct / float(num_instance)

    def get_recall(self):
        num_instance = self.dataset.get_instance_size()
        rc_per_instance = []
        for inst_id in range(1, num_instance + 1):
            inst = self.dataset.get_instance(inst_id)
            label_ids = inst.get_label_id_list()
            true_positive = 0.0
            positive = 0.0
            for label_id in label_ids:
                integrated_label = inst.get_integrated_label(label_id)
                true_label = inst.get_true_label(label_id)
                if (true_label.val == 2):
                    positive += 1.0
                    if (integrated_label.val == 2):
                        true_positive += 1.0
            rc_per_instance.append(true_positive/ positive)
        return sum(rc_per_instance) / float(num_instance)

    def get_precision(self):
        num_instance = self.dataset.get_instance_size()
        pc_per_instance = []
        for inst_id in range(1, num_instance + 1):
            inst = self.dataset.get_instance(inst_id)
            label_ids = inst.get_label_id_list()
            true_positive = 0.0
            positive = 0.0
            for label_id in label_ids:
                integrated_label = inst.get_integrated_label(label_id)
                true_label = inst.get_true_label(label_id)
                if (integrated_label.val == 2):
                    positive += 1.0
                    if (true_label.val == 2):
                        true_positive += 1.0
            pc_per_instance.append(true_positive/ positive)
        return sum(pc_per_instance) / float(num_instance)

    def get_f1_score(self):
        num_instance = self.dataset.get_instance_size()
        f1_per_instance = []
        for inst_id in range(1, num_instance + 1):
            inst = self.dataset.get_instance(inst_id)
            label_ids = inst.get_label_id_list()
            joint_positive = 0.0
            y_positive = 0.0
            z_positive = 0.0
            for label_id in label_ids:
                integrated_label = inst.get_integrated_label(label_id)
                true_label = inst.get_true_label(label_id)
                if ((integrated_label.val == 2) and (true_label.val == 2)):
                    joint_positive += 1.0
                if (integrated_label.val == 2):
                    z_positive += 1.0
                if (true_label.val == 2):
                    y_positive += 1.0
            f1_per_instance.append((2*joint_positive) / (z_positive + y_positive))
        return sum(f1_per_instance) / float(num_instance)
