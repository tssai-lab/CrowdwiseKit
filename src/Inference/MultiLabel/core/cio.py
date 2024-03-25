# -*- coding: utf-8 -*-

from Inference.MultiLabel.core import data

def load_file(resp_path, gold_path = None):
    """
    load response file and gold file to create data set
    :param resp_path: response file
    :param gold_path: gold file [optional]
    :return:
    """
    dataset = data.Dataset()
    has_gold_file = False
    if gold_path != None:
        gold_file = open(gold_path)
        for line in gold_file:
            strs = line.split()
            if len(strs) <= 1:
                gold_file.close()
                print('Error formatted gold file', end='\n')
                return None

            inst_id = dataset.fetch_instance_id_by_name(strs[0])
            inst = dataset.get_instance(inst_id)
            if inst == None:
                inst = data.Instance(inst_id)
                dataset.add_instance(inst)

            label_id =  1
            label_val = 0
            if (len(strs) + 1) == data.Label.SINGLE_LABLE:
                label_id = dataset.fetch_label_id_by_name('SINGLE_LABEL')
                label_val = dataset.fetch_label_val_id_by_name(label_id, strs[1])  # label id is 1
            if (len(strs) + 1) == data.Label.MULTI_LABEL:
                label_id = dataset.fetch_label_id_by_name(strs[1])
                label_val = dataset.fetch_label_val_id_by_name(label_id, strs[2])
            true_label = inst.get_true_label(label_id)
            if true_label == None:
                true_label = data.Label(label_id)
                true_label.inst_id = inst.id
                true_label.worker_id = data.Worker.GOLD
                inst.add_true_label(true_label)
            true_label.val = label_val

        has_gold_file = True
        gold_file.close()

    resp_file = open(resp_path)
    for line in resp_file:
        strs = line.split()
        if len(strs) <= 2:
            resp_file.close()
            print('Error formatted response file', end='\n')
            return None

        worker_id = dataset.fetch_worker_id_by_name(strs[0])
        worker = dataset.get_worker(worker_id)
        if worker == None:
            worker = data.Worker(worker_id)
            dataset.add_worker(worker)

        inst_id = data.Instance.INVALID_ID
        if has_gold_file == True:
            inst_id = dataset.get_instance_id_by_name(strs[1])
            if inst_id == data.Instance.INVALID_ID:
                print('Warning find an instance ' + strs[1] +' not in gold file, skip it', end='\n')
                continue
        else:
            inst_id = dataset.fetch_instance_id_by_name(strs[1])

        label_id = 1 #default label _id for single labeling
        label_val = 0
        if  len(strs) == data.Label.SINGLE_LABLE:
            if has_gold_file == False:
                label_id = dataset.fetch_label_id_by_name('SINGLE_LABEL')
            label_val = dataset.fetch_label_val_id_by_name(label_id, strs[2])  # label id is 1
        if  len(strs) == data.Label.MULTI_LABEL:
            if has_gold_file == True:
                label_id = dataset.get_label_id_by_name(strs[2])
                if label_id == data.Label.INVALID_ID:
                    print('Warning find a label name ' + strs[2] + ' not in gold file, skip it', end='\n')
                    continue
            else:
                label_id = dataset.fetch_label_id_by_name(strs[2])
            label_val = dataset.fetch_label_val_id_by_name(label_id, strs[3])

        inst = dataset.get_instance(inst_id)
        if inst == None:
            inst = data.Instance(inst_id)
            dataset.add_instance(inst)
        label_info = (label_id, inst.id, worker.id)
        label = worker.get_label(label_info)
        if label == None:
            label = data.Label(label_id)
            label.inst_id = inst.id
            label.worker_id = worker.id
            worker.add_label(label)
        label.val = label_val
        inst.add_noisy_label(label)
        
    resp_file.close()
    dataset.label_info_confirm()
    dataset.create_from_files = True
    return dataset


def save_file(dataset, resp_path, gold_path = None):
    """
    save a dataset to response and gold files
    :param dataset: dataset to be saved
    :param resp_path: response file
    :param gold_path: gold file [optional]
    :return:
    """
    multi_class = dataset.is_multi_class()

    if gold_path != None:
        gold_file = open(gold_path, 'w')
        inst_num = dataset.get_instance_size()
        for id in range(1, inst_num + 1):
            inst = dataset.get_instance(id)
            true_label_set = inst.get_true_label_set()
            if len(true_label_set) == 1:
                gold_file.write(str(inst.id) + '\t' + str(inst.get_true_label(1).val)+'\n')
            else:
                for (k, v) in true_label_set.items():
                    gold_file.write(str(inst.id) + '\t' + str(k) +'\t' + str(v.val) + '\n')
        gold_file.close()

    resp_file = open(resp_path, 'w')
    worker_num = dataset.get_worker_size()
    for id in range (1, worker_num + 1):
        worker = dataset.get_worker(id)
        labels = worker.get_label_list()
        for label in labels:
            if multi_class == True:
                resp_file.write(str(worker.id)+ '\t' + str(label.inst_id) + '\t' + str(label.id) + '\t' + str(label.val) + '\n')
            else:
                resp_file.write(str(worker.id) + '\t' + str(label.inst_id) + '\t' + str(label.val) + '\n')

    resp_file.close()

def save_map_file(dataset, map_path):
    if dataset.create_from_files:
        map_file = open(map_path, 'w')
        map_file.write('Label id mapping: \n')
        for label_id in range(1, dataset.get_label_id_size() + 1):
            map_file.write(dataset.get_label_name_by_id(label_id) + '\t\t\t(id='+ str(label_id) + ')\n')
        map_file.write('\nLabel value mapping: \n')
        for label_id in range(1, dataset.get_label_id_size() + 1):
            label_name =  dataset.get_label_name_by_id(label_id)
            for value_id in range(1,  dataset.get_label_val_size(label_id)+1):
                 map_file.write(label_name + '(id='+ str(label_id)+')\t\t\t' + \
                                dataset. get_label_val_name_by_id(label_id, value_id) + '(id='+str(value_id) +')\n')
        map_file.write('\nWorker mapping: \n')
        for worker_id in range(1, dataset.get_worker_size() + 1):
            worker_name = dataset.get_worker_name_by_id(worker_id)
            map_file.write(worker_name + '\t\t\t(id='+ str(worker_id) + ')\n')
        map_file.write('\nInstance mapping: \n')
        for inst_id in range(1, dataset.get_instance_size() + 1):
            inst_name = dataset.get_instance_name_by_id(inst_id)
            map_file.write(inst_name + '\t\t\t(id=' + str(inst_id) + ')\n')
        map_file.close()
