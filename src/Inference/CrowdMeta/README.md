# CrowdMeta

If you use CrowdMeta, please cite the following paper:

```
@article{zhang2023crowdmeta,
	title={CrowdMeta: Crowdsourcing Truth Inference with Meta-Knowledge Transfer},
	author={Zhang, Jing and Xu, Sunyue and Sheng, Victor S},
	journal={Pattern Recognition},
	pages={109525},
	year={2023},
	publisher={Elsevier}
}
```

## Datasets

### 1. Dataset MiniImageNet
The MiniImageNet dataset was derived from the ImageNet dataset. ImageNet is a well-known large-scale visual dataset, which was established to facilitate CV research. Training the ImageNet dataset consumes a lot of computing resources. ImageNet annotates over 14 million images and provides bounding boxes for at least 1 million images. ImageNet contains more than 20,000 categories and each category has no less than 500 images.

Training so many images consumes a lot of resources, so in 2016, Oriol Vinyals et al. in Google DeepMind team extracted the MiniImageNet dataset based on ImageNet. If you use the MiniImageNet dataset in your work, please cite the following paper:

```
@inproceedings{Vinyals2016,
	author = {Vinyals, Oriol and Blundell, Charles and Lillicrap, Timothy and Kavukcuoglu, Koray and Wierstra, Daan},
	title = {Matching networks for one shot learning},
	year = {2016},
	booktitle = {NIPS},
	pages = {3637--3645},
	numpages = {9},
}
```

MiniImagenet is sized 2.86GB. File structure is as follows：

 root/ &nbsp;  
 &emsp;  |- images/  
 &emsp; &emsp; |- n0153282900000005.jpg   
 &emsp; &emsp; |- n0153282900000006.jpg  
 &emsp; &emsp; |- …  
 &emsp; |- train.csv   
 &emsp; |- test.csv  
 &emsp; |- val.csv  

The MiniImagenet dataset can be downloaded from [github:mini-imagenet](https://github.com/yaoyao-liu/mini-imagenet-tools).

### 2. Dataset Omniglot
The Omniglot data is a full-language text dataset that contains different alphabets of various languages, such as Japanese hiragana, Japanese katakana, Korean vowels and consonants, the most common Latin alphabet, etc. Omniglot contains a total of 50 alphabets in different languages, each alphabet contains different characters, a total of 1623 characters, and each character is written by 20 different people. The Omniglot dataset contains 1623 classes and each class has 20 training data. If you use the Omniglot dataset in your work, please cite the following paper:
```
@article{Omniglot,
	author = {Brenden M. Lake  and Ruslan Salakhutdinov  and Joshua B. Tenenbaum },
	title = {Human-level concept learning through probabilistic program induction},
	journal = {Science},
	volume = {350},
	number = {6266},
	pages = {1332-1338},
	year = {2015},
	doi = {10.1126/science.aab3050},
}
```
The Omniglot dataset can be loaded from [github:omniglot](https://github.com/brendenlake/omniglot). The download warehouse provides APIs for python and matlab respectively. Download and unzip `images_background.zip` and `images_evaluation.zip` in the python directory, which are the training data and test data.

## Run
The functions of source code:

learner.py  network building  
meta.py  network training  
MiniImagenet.py  miniimagenet  data loaded  
miniimagenet_train.py  training and fine-tuning  
omniglot.py  omniglot  data loaded  
omniglot_train.py  training and fine-tuning  
meanstd.py  feature distribution extraction  
infer.py  truth inference  

Note: Because the structure of the Miniimagenet dataset is slightly different from that of the Omniglot dataset, the data loading is slightly different. If you want to use other datasets, it is recommended to preprocess the data file structure to the same structure as the Miniimagenet dataset. There is essentially no difference between the training and finetuning files `miniimagenet_train.py` and `omniglot_train.py`, except for the data loading.

### 1. pre-training+finetuning
run `miniimagenet_train.py` and obtain high-order feature representations.
```bash
#python method.py
python miniimagenet_train.py
```
### 2. feature distribution extraction
run `meanstd.py` to extract the feature distribution of the target task. The target task dataset.txt stores the address of the task image.
```bash
#python method.py <file of target tasks> <file of feature distributions>
python meanstd.py dataset.txt distribution.txt
```
### 3. truth inference
run `infer.py` to obtain a file storing the final integrated labels and accuracy.
```bash
#python method.py <.resp file> <.gold file> <file of high-order features> <file of feature distributions>
python infer.py label.resp truth.gold feature.attr distribution.txt
```