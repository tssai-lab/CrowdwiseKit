The source code of the paper Learning from Crowds with Multiple Feature Dynamic Fusion-Based Annotation Generation submitted to TPAMI.


# MFDFAGen-Net
This repository contains the implementation for the paper:  
**Learning from Crowds with Multiple Feature Dynamic Fusion-Based Annotation Generation**

## Environment Setup
Install required dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation
1. Create a `data` directory in the project root
2. Place datasets in the `data` folder

Preprocessed datasets are available at:  
[GitHub Dataset Repository](https://github.com/stop-hand/dataset-about-crowdsouring)

### Official Dataset Sources
| Dataset  | Source |
|----------|--------|
| LabelMe  | [Deep learning from crowds](https://www.cs.ubc.ca/~murphyk/Software/crowd/crowd.html) |
| Music    | [UCI Repository](http://archive.ics.uci.edu/ml/) <br> [AMILab](http://amilab.dei.uc.pt/fmpr/software/7) |
| Cifar10N | [NoisyLabels.com](http://noisylabels.com) <br> [Starter Code](https://github.com/UCSC-REAL/cifar-10-100n) |

## Execution
Run the appropriate training script for your dataset:
```bash
python train_$dataset_name$.py
```
> Replace `$dataset_name$` with your target dataset (e.g., `labelme`, `music`, `cifar10n`)
