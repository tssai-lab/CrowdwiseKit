# Multi-Label inference algorithms
Here are some truth inference algorithms for multi-label crowdsourcing annotation tasks.

## Identifiers for Instances, Workers, Labels and Class Values
All internal identifiers for instances, workers, labels and class values start from one.\
Zero means an empty annotation.

## Input Files
1. **.resp** file: the file containing the responses of crowdsourced workers to the questions
- format for *single-label* `worker_id` `instance_id` `label_value`
- format for *multi-label* &nbsp; `worker_id` `instance_id` `label_id` `label_value`
2. **.gold** file: the file containing the ground truth for performance evaluation
- format for *single-label* `instance_id` `label_value`
- format for *multi-label*  &nbsp; `instance_id` `label_id` `label_value`

## Multi-Label Truth Inference Algorithms
1. **Multi-Class Multi-Label Independent (MCMLI)**
2. **Multi-Class Multi-Label Dependent (MCMLD)**
3. **Multi-Class Multi-Label Independent One-Coin model (MCMLI-OC)**
4. **Multi-Class Multi-Label Dependent One-Coin model (MCMLD-OC)**
5. **Majority Voting (MV, extended to multi-label scenario)**
6. **Dawid and Skene's model (DS, extended to mulit-label scenario)**
7. **Independent Bayesian Classifier Combination (iBCC, extended to mulit-label scenario)**
8. **Multi-Class One-Coin model (MCOC, extended to multi-label scenario)**

