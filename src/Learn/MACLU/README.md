# MALC Description
This document will introduce the special specifications and requirements for multi-label active learning in the input file of this algorithm, as well as the definition and usage of related functions in this algorithm. The general format of the input file is in accordance with the requirements of CrowdKit.

## Input dataset file specification
This section defines the MALC algorithm input dataset specification:

- This algorithm is designed for truth inference and classification in a multi-label binary classification scenario, so the category value under each label is selected from {1,2}, and when the category value is 0, it means that the label of the sample has not been labeled .
- The input dataset consists of a small portion of labeled data and a large portion of unlabeled data. And assume that in the labeled data, each label of the sample has at least one label.
- The input of the algorithm includes the path of the .resp file and the .attr file, and the path of the .gold file is an optional parameter.

## Algorithm Definition
### activelearnerMACLU(in_resp_path,in_attr_path,in_gold_path,instance_num,worker_num,label_num)
Input:
in_resp_path: work annotation path
in_attr_path: instance feature path
in_gold_path: instance true label path
instance_num: the total number of instances
worker_num: the total number of workers
label_num: the total number of labels
return an instance of class activelearnerMACLU.

### model.initialize()
Initialization function for processing input files and initializing model parameters.

### model.infer()
Used to infer aggregated labels for labeled samples; for unlabeled samples, train a multi-label classifier.

### model.select_next()
Returns a sample-label-worker tuple to be queried in the next step. At the same time, it also provides the interface of each individual active learning strategy, that is, the sample selection strategy select_next_instance(), the label selection strategy select_next_label() and the worker selection strategy select_next_worker().

### model.update(anno)
Input:
anno: After obtaining the sample-label-worker set by the select_next() method, the selected worker provides the labeled value of the selected label for the selected sample.
Used to update the model data after the active learning strategy selects the appropriate sample-label-worker triplet and obtains the labels.

### print_aggregate_accuracy()
Print label integration accuracy.

### print_predict_accuracy()
Print prediction accuracy.

### print_total_accuracy()
Print total accuracy of the algorithm (including label integration and prediction).
