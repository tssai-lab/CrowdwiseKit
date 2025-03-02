# CrowdwiseKit
A Tool Kit for Crowdsourcing Learning.

This project was created by the TSSAI (Trustworthy Secure and Safe Artificial Intelligence) Lab led by Professor Jing Zhang at Southeast University.

The package includes three parts: **Inference**, **Learn**, and **Incentive**.

## Inference
The ground truth inference (label aggregation or label integration) methods for crowdsourced annotation.

- CUBAM: Ref.  "The multidimensional wisdom of crowds. NIPS-2010."

- CrowdMKT: Ref. "Crowdsourcing with multiple-source knowledge transfer. IJCAI-2020."

- CrowdMeta: A truth inference model that uses meta knowledge transfer. Ref. "Crowdmeta: Crowdsourcing truth inference with meta-Knowledge transfer. Pattern Recognition. 2023."

- CrowdTrU: Ref. "Active learning for crowdsourcing using knowledge transfer. AAAI-2014."

- GLAD: A classical truth inference model that model instance difficulties. Ref. "Whose vote should count more: Optimal integration of labels from labelers of unknown expertise. NIPS-2009." 

- Multilabel: A collection of multi-label inference methods
	- Multi-Class Multi-Label Independent (MCMLI)
	- Multi-Class Multi-Label Dependent (MCMLD)
	- Multi-Class Multi-Label Independent One-Coin model (MCMLI-OC)
	- Multi-Class Multi-Label Dependent One-Coin model (MCMLD-OC)
	- Majority Voting (MV, extended to multi-label scenario)
	- Dawid and Skene's model (DS, extended to mulit-label scenario)
	- Independent Bayesian Classifier Combination (iBCC, extended to mulit-label scenario)
	- Multi-Class One-Coin model (MCOC, extended to multi-label scenario)

- RY: A classical truth inference model derived from the Dawid and Skene's model. Ref. "Learning from crowds. Journal of Machine Learning Research. 2010."

- Yan: Ref. "Modeling annotator expertise: Learning when every body knows a bit of something. Proc. 13 Int. Conf. Artif. Intell. Stat., 2010."

- ZenCrowd: "Zencrowd: leveraging probabilistic reasoning and crowdsourcing techniques for large-scale entity linking. WWW-2012."

## learn
Learn prediction models from the crowdsourced labeled data. The part mainly focuses on end-to-end crowdsourcing learning models.

- CGNNAT: An end-to-end crowdsourcing learning model using graph neural networks with an attention mechanism. Ref. " Learning from crowds using graph neural networks with attention mechanism. IEEE Transactions on Big Data. 2025."

- MACLU:  Multi-label active learning from crowds. Ref. " Multi-label active learning from crowds for secure IIoT. Ad Hoc Networks. 2021."

## Incentive
The incentive method for federated crowdsourcing.

## Other Useful Information
- [Ceka](https://ceka.sourceforge.net/): An early Java version for Crowdsourcing Learning, also created by Professor Jing Zhang.
- 
