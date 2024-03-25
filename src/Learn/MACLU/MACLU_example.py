import numpy as np

from MACLU import activelearnerMACLU

in_resp_partial_path = 'dataset/scene-partial-train_1.resp'
in_resp_path = 'dataset/scene-train_1.resp'
in_gold_path = 'dataset/scene-train.gold'
in_attr_path = 'dataset/scene-train.attr'

out_log_path = 'results/output.log'

instance_num = 1211
worker_num = 10
label_num = 6

instance_matrix = np.zeros((instance_num, worker_num, label_num))
resp_f = open(in_resp_path)
for line in resp_f:
    strs = line.split()
    if len(strs) <= 3:
        resp_f.close()
        print('Error formatted response file', end='\n')
    instance_matrix[int(strs[1]) - 1][int(strs[0]) - 1][int(strs[2]) - 1] = int(strs[3])
resp_f.close()

out_log = open(out_log_path, 'w')

almalc = activelearnerMACLU(in_resp_partial_path, in_attr_path, instance_num, worker_num, label_num, in_gold_path)
almalc.initialize()

qcnt = 1
while qcnt <= almalc.query_num:

    almalc.infer()
    (instance_id, label_id, worker_id) = almalc.select_next()
    print('The selected triplet is (' + str(instance_id) + ',' + str(label_id) + ',' + str(worker_id) + ')')
    get_value = instance_matrix[instance_id-1][worker_id-1][label_id-1]
    almalc.update(get_value)
    out_log.write('The selected triplet is (' + str(instance_id) + ',' + str(label_id) + ',' + str(worker_id))
    out_log.write('\n')
    if qcnt == 1:
        print('The accuracy of round ', qcnt, 'is ', almalc.print_predict_accuracy())
        out_log.write('The predict accuracy of round ' + str(qcnt) + ' is ' + str(almalc.print_predict_accuracy()))
        out_log.write('\n')
    if qcnt % 50 == 0:
        print('The accuracy of round ', qcnt, 'is ', almalc.print_predict_accuracy())
        out_log.write('The predict accuracy of round ' + str(qcnt) + ' is ' + str(almalc.print_predict_accuracy()))
        out_log.write('\n')
    qcnt += 1

out_log.close()
