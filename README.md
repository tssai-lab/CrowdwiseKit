# CrowdwiseKit
A Tool Kit for Crowdsourcing Learning.

This project was created by the TSSAI (Trustworthy Secure and Safe Artificial Intelligence) Lab led by Professor Jing Zhang at Southeast University.

The package includes three parts: **Inference**, **Learn**, and **Incentive**.

## Inference
The ground truth inference (label aggregation or label integration) methods for crowdsourced annotation.

- **CUBAM**: Ref.  "The multidimensional wisdom of crowds. NIPS-2010."

- **CrowdMKT**: Ref. "Crowdsourcing with multiple-source knowledge transfer. IJCAI-2020."

- **CrowdMeta**: A truth inference model that uses meta knowledge transfer. Ref. "Crowdmeta: Crowdsourcing truth inference with meta-Knowledge transfer. Pattern Recognition. 2023."

- **CrowdTrU**: Ref. "Active learning for crowdsourcing using knowledge transfer. AAAI-2014."

- **GLAD**: A classical truth inference model that model instance difficulties. Ref. "Whose vote should count more: Optimal integration of labels from labelers of unknown expertise. NIPS-2009." 

- **Multilabel**: A collection of multi-label inference methods
	- **MCMLI**: Multi-Class Multi-Label Independent 
	- **MCMLD**: Multi-Class Multi-Label Dependent
	- **MCMLI-OC**: Multi-Class Multi-Label Independent One-Coin model
	- **MCMLD-OC**: Multi-Class Multi-Label Dependent One-Coin model
	- **MV**: Majority Voting (extended to multi-label scenario)
	- **DS**: Dawid and Skene's model (extended to multi-label scenario)
	- **iBCC**: Independent Bayesian Classifier Combination (extended to multi-label scenario)
	- **MCOC**: Multi-Class One-Coin model (extended to multi-label scenario)

- **LFC**:Supervised learning with subjective labels provided by multiple annotators and evaluation of different experts to estimate the actual hidden labels. Ref. "Learning From Crowds. Journal of Machine Learning Research. 2010."

- **EBCC**:An model is proposed to effectively model worker correlation in crowdsourced annotation by mixing mean field variational inference and intra-class reliability. Ref. "Exploiting worker correlation for label aggregation in crowdsourcing. International conference on machine learning. 2019."

- **IWMV**:An Iterative Weighted Majority Voting method that theoretically optimizes the upper bound of the average error rate of weighted majority voting. Ref. "Error rate bounds and iterative weighted majority voting for crowdsourcing. arXiv:1411.4086. 2014."

- **RY**: A classical truth inference model derived from the Dawid and Skene's model. Ref. "Learning from crowds. Journal of Machine Learning Research. 2010."

- **Yan**: Ref. "Modeling annotator expertise: Learning when every body knows a bit of something. Proc. 13 Int. Conf. Artif. Intell. Stat., 2010."

- **ZenCrowd**: Ref. "Zencrowd: leveraging probabilistic reasoning and crowdsourcing techniques for large-scale entity linking. WWW-2012."

## learn
Learn prediction models from the crowdsourced labeled data. The part mainly focuses on end-to-end crowdsourcing learning models.

- **CrowdLayer**:A novel general-purpose crowd layer to train deep neural networks end-to-end. Ref. "Deep Learning from Crowds. AAAI-2018."
  
- **CoNAL**:Crowdsourcing model by an end-to-end learning solution with two types of noise adaptation layers. Ref. "Learning from Crowds by Modeling Common Confusions. AAAI-2021."
  
- **CGNNAT**: An end-to-end crowdsourcing learning model using graph neural networks with an attention mechanism. Ref. " Learning from crowds using graph neural networks with attention mechanism. IEEE Transactions on Big Data. 2025."
- **CCC**:Coupled Confusion Correction, where two models are simultaneously trained to correct the confusion matrices learned by each other. Via bi-level optimization, the confusion matrices learned by one model can be corrected by the distilled data from the other. Ref. "Coupled Confusion Correction: Learning from Crowds with Sparse Annotations. AAAI-2024."
  
- **MACLU**:  Multi-label active learning from crowds. Ref. " Multi-label active learning from crowds for secure IIoT. Ad Hoc Networks. 2021."
  
- **TAIDTM**:Estimating annotator- and instance-dependent transfer matrices via knowledge transfer to address the label noise problem when learning from crowdsourced data. Ref. "Transferring Annotator- and Instance-Dependent Transition Matrix for Learning From Crowds. IEEE Transactions on Pattern Analysis and Machine Intelligence. 2024."

## Incentive
The incentive method for federated crowdsourcing.

- **TsIFedCrowd**: Ref. "Timeliness-selective incentive federated crowdsourcing. ICWS-2024."

- **TiFedCrowd**: Ref. "TiFedCrowd: Federated crowdsourcing with time-controlled incentive. IEEE Transactions on Emerging Topics in Computational Intelligence. 2025."

## zone-ex (zone external) *[Independent Projects]*
We have collected some code written by other researchers, which is put in directory [zone-ex](https://github.com/tssai-lab/CrowdwiseKit/tree/main/zone-ex). We do not guarantee that these programs will work well.
- **GLAD**: The source code was created by Whitehill. We modified it and created an executive file that can run on Windows. [*Paper: Whitehill J., Wu T. F., Bergsma J., Movellan J., & Ruvolo P. (2009). Whose vote should count more: Optimal integration of labels from labelers of unknown expertise. NIPS, 22, 2035-2043.*]
- **crowd_tool_nips2012liu.zip**: The source code was created by Qiang Liu, written in MATLAB. [*Paper: Liu Q., Peng J., & Ihler A. T. (2012). Variational inference for crowdsourcing. In NIPS, 692-700.*]
- **spectral_ds_jmlr2016zhang.zip**: The source code was created by Yuchen Zhang, written in MATLAD, implementing the DS model that is initialized with Spectral methods. [*Paper: Zhang Y., Chen X., Zhou D., & Jordan M. I. (2016). Spectral methods meet EM: A provably optimal algorithm for crowdsourcing. The Journal of Machine Learning Research, 17(1), 3537-3580.*]
- **code_icml2017chen.zip**: The source code was created by Guangyong Chen, written in MATLAB, implementing the aggregation for ordinal labels. [*Paper: Chen G., Zhang S., Lin D., Huang H., & Heng P. A. (2017). Learning to aggregate ordinal labels by maximizing separating width. In ICML, 787-796.*]
- **truth_inference_crowd_vldb2017zheng.zip**: It collected and implemented many truth inference algorithms by Yundian Zheng, et al. The truth inference algorithms include MV, GLAD, DS, Minimax, BCC, cBCC, LFC, CATD, PM, MultiD, KOS, VI-BP, VI-MF, LFC_N, Mean, and Median. They were implemented or encapsulated in Python. More information can be found [here](https://zhydhkcws.github.io/crowd_truth_inference/index.html). [*Paper: Zheng Y., Li G., Li Y., Shan C., & Cheng R. (2017). Truth inference in crowdsourcing: Is the problem solved?. Proceedings of the VLDB Endowment, 10(5), 541-552.*]

	**NOTE: If you use the above source code, please cite the corresponding papers.**

## Datasets
**More real-world raw crowdsourcing datasets can be found [here](https://wocshare.sourceforge.io/).**

## Other Useful Information
- [Ceka](https://ceka.sourceforge.net/): An early Java version for Crowdsourcing Learning, also created by Professor Jing Zhang.
- [SQUARE](https://ai.ischool.utexas.edu/square/): A benchmark for comparative evaluation of consensus methods for crowdsourcing.
- [BATC](https://code.google.com/archive/p/benchmarkcrowd/): Benchmark for aggregate techniques in crowdsourcing.
