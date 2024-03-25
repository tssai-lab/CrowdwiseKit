# -*- coding: utf-8 -*-
import os
from Inference.MultiLabel.core import cio
from Inference.MultiLabel.core import data
from Inference.MultiLabel.core import perf
from Inference.MultiLabel.core import utils

from Inference.MultiLabel import ds
from Inference.MultiLabel import mv
from Inference.MultiLabel import mcmli
from Inference.MultiLabel import mcmld
from Inference.MultiLabel import mcoc
from Inference.MultiLabel import ibcc

#in_resp_path = 'D:/Github/datasets/aircrowd6.response.txt'
#in_gold_path = 'D:/Github/datasets/aircrowd6.gold.txt'

#in_resp_path = 'D:/Github/datasets/rte.response.txt'
#in_gold_path = 'D:/Github/datasets/rte.gold.txt'

#in_resp_path = 'D:/Github/datasets/valence5.response.txt'
#in_gold_path = 'D:/Github/datasets/valence5.gold.txt'

#in_resp_path = 'D:/Github/datasets/valence7.response.txt'
#in_gold_path = 'D:/Github/datasets/valence7.gold.txt'

#in_resp_path = 'D:/Github/datasets/synth.resp'
#in_gold_path = 'D:/Github/datasets/synth.gold'

#in_resp_path = 'D:/Github/datasets/yeast/yeast-train.resp'
#in_gold_path = 'D:/Github/datasets/yeast/yeast-train.gold'

in_resp_path = 'D:/Github/datasets/emotions/emotions-train.resp'
in_gold_path = 'D:/Github/datasets/emotions/emotions-train.gold'

#in_resp_path = 'D:/zcrom/Output/affective-ml.resp'
#in_gold_path = 'D:/zcrom/Output/affective-ml.gold'

#in_resp_path = 'D:/zcrom/Output/affective-mlk4.resp'
#in_gold_path = 'D:/zcrom/Output/affective-mlk4.gold'

#out_resp_path = 'D:/Github/datasets/aircrowd6.resp'
#out_gold_path = 'D:/Github/datasets/aircrowd6.gold'

dataset = cio.load_file(in_resp_path, in_gold_path)
map_path  = os.path.splitext(in_resp_path)[0] + '.map'
cio.save_map_file(dataset, map_path)

coefficient = dataset.get_num_explained_components_by_PCA(0.8)
print('coefficient = ' + str(coefficient))

# core.cio.save_file(dataset, out_resp_path, out_gold_path)
maxround = 20
soft = True

mv = mv.MVModel()
mv.infer(dataset)
eval = perf.Evaluation(dataset)
num_label = dataset.get_label_id_size()
for label_id in range(1, num_label + 1):
    print('MV acc on label (' + str(label_id) +'): '+ str(eval.get_accuracy_on_label(label_id)))
print('MV acc: ' + str(eval.get_accuracy()) + ' subset acc: ' + str(eval.get_subset_accuracy()))

#R=3
#ocld = ocld.OCLDModel(R, maxround)
#ocld.set_converge_rate(0.005)
#ocld.infer(dataset, soft)


"""
ibcc = inference.ibcc.IBCCModel()
ibcc.sampling_infer(dataset, 10, 5)
eval = core.perf.Evaluation(dataset)
num_label = dataset.get_label_id_size()
for label_id in range(1, num_label + 1):
    print('IBCC acc on label (' + str(label_id) +'): '+ str(eval.get_accuracy_on_label(label_id)))
print('IBCC acc: ' + str(eval.get_accuracy()) + ' subset acc: ' + str(eval.get_subset_accuracy()))

ds = inference.ds.DSModel(maxround)
ds.infer(dataset, soft)
eval = core.perf.Evaluation(dataset)
num_label = dataset.get_label_id_size()
for label_id in range(1, num_label + 1):
    print('DS acc on label (' + str(label_id) +'): '+ str(eval.get_accuracy_on_label(label_id)))
print('DS acc: ' + str(eval.get_accuracy()) + ' subset acc: ' + str(eval.get_subset_accuracy()))

mcoc = inference.mcoc.MCOCModel(maxround)
mcoc.infer(dataset, soft)
eval = core.perf.Evaluation(dataset)
num_label = dataset.get_label_id_size()
for label_id in range(1, num_label + 1):
    print('MCOC acc on label (' + str(label_id) +'): '+ str(eval.get_accuracy_on_label(label_id)))
print('MCOC acc: ' + str(eval.get_accuracy()) + ' subset acc: ' + str(eval.get_subset_accuracy()))

mcmli = inference.mcmli.MCMLIModel(maxround)
mcmli.infer(dataset, soft)
eval = core.perf.Evaluation(dataset)
num_label = dataset.get_label_id_size()
for label_id in range(1, num_label + 1):
    print('MCMLI acc on label (' + str(label_id) +'): '+ str(eval.get_accuracy_on_label(label_id)))
print('MCMLI acc: ' + str(eval.get_accuracy()) + ' subset acc: ' + str(eval.get_subset_accuracy()))

R=4
mcmld = inference.mcmld.MCMLDModel(R, 20)
omega = [None]
rlist = core.utils.gen_rand_sum_one(R)
print(rlist)
for r in rlist:
    omega.append(r)
mcmld.set_omega(omega)
mcmld.set_converge_rate(0.005)
mcmld.infer(dataset, soft)
eval = core.perf.Evaluation(dataset)
num_label = dataset.get_label_id_size()
for label_id in range(1, num_label + 1):
    print('MCMLD acc on label (' + str(label_id) +'): '+ str(eval.get_accuracy_on_label(label_id)))
print('MCMLD acc: ' + str(eval.get_accuracy()) + ' subset acc: ' + str(eval.get_subset_accuracy()))

# doc = inference.doc.DOCModel(R, maxround)
# doc.set_omega(omega)
# doc.set_converge_rate(0.005)
# doc.infer(dataset, soft)
# eval = core.perf.Evaluation(dataset)
# num_label = dataset.get_label_id_size()
# for label_id in range(1, num_label + 1):
#    print('DOC acc on label (' + str(label_id) +'): '+ str(eval.get_accuracy_on_label(label_id)))
# print('DOC acc: ' + str(eval.get_accuracy()) + ' subset acc: ' + str(eval.get_subset_accuracy()))
"""
