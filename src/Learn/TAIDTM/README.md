# <a href="https://arxiv.org/abs/2306.03116" target="_blank"> Transferring Annotator- and Instance-dependent Transition Matrix for Learning from Crowds </a> - Official PyTorch Code (IEEE TPAMI 2024)

### Abstract:
Learning from crowds describes that the annotations of training data are obtained with crowd-sourcing services. Multiple annotators each complete their own small part of the annotations, where labeling mistakes that depend on annotators occur frequently. Modeling the label-noise generation process by the noise transition matrix is a power tool to tackle the label noise. In real-world crowd-sourcing scenarios, noise transition matrices are both annotator- and instance-dependent. However, due to the high complexity of annotator- and instance-dependent transition matrices (AIDTM), annotation sparsity, which means each annotator only labels a little part of instances, makes modeling AIDTM very challenging. Prior works simplify the problem by assuming the transition matrix is instance-independent or using simple parametric ways, which lose modeling generality. Motivated by this, we target a more realistic problem, estimating general AIDTM in practice. Without losing modeling generality, we parameterize AIDTM with deep neural networks. To alleviate the modeling challenge, we suppose every annotator shares its noise pattern with similar annotators, and estimate AIDTM via knowledge transfer. We hence first model the mixture of noise patterns by all annotators, and then transfer this modeling to individual annotators. Furthermore, considering that the transfer from the mixture of noise patterns to individuals may cause two annotators with highly different noise generations to perturb each other, we employ the knowledge transfer between identified neighboring annotators to calibrate the modeling. Theoretical analyses are derived to demonstrate that both the knowledge transfer from global to individuals and the knowledge transfer between neighboring individuals can help model general AIDTM. Experiments confirm the superiority of the proposed approach on synthetic and real-world crowd-sourcing data.


### Running the code on Fashion-MNIST Dataset:
Put Fashion-MNIST dataset in data folder, and run the code by using the provided script.


### Citation:
If you find the code useful in your research, please consider citing our paper:

```
 @article{Li2024TAIDTM,
  title={Transferring Annotator-and Instance-dependent Transition Matrix for Learning from Crowds},
  author={Li, Shikun and Xia, Xiaobo and Deng, Jiankang and Ge, Shiming and Liu, Tongliang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year={2024}
}
```
