# Common Noise Adaptation Layers
Code for AAAI 2021 long paper **Learning from Crowds by Modeling Common Confusions**.

## Dependencies
* Python 3.5+
* PyTorch 1.2.0

## Datasets
Download the real-world dataset [LabelMe](http://fprodrigues.com//deep_LabelMe.tar.gz) and [Music](http://fprodrigues.com//mturk-datasets.tar.gz). Please use the pretrained features in the *prepared* folder and put them into the *data* folder. 

## Usage
Run the model with default settings
```
python main.py
```

## Citation
Please cite our work if you find it useful to your research
 ```
@article{chu2020learning,
  title={Learning from Crowds by Modeling Common Confusions},
  author={Chu, Zhendong and Ma, Jing and Wang, Hongning},
  journal={arXiv preprint arXiv:2012.13052},
  year={2020}
}
 ```
