# RLNet
## Overview
Overall Architecture of RouterFormer.
<img width="5123" height="2665" alt="图片15" src="https://github.com/user-attachments/assets/354b45d7-7d1c-4259-86fd-c608d27425b5" />


## Installation
```
# first installation  method
conda env create -f environment.yml
conda activate routerformer
python setup.py develop

# second installation method
git clone https://github.com/chengtan9907/OpenSTL
cd OpenSTL
conda env create -f environment.yml
conda activate OpenSTL
python setup.py develop
```

## Dataset
### SiChuan Dataset
We are not authorized to release this dataset.
### HKO-7 Dataset
It can be obtained via the following address: https://github.com/sxjscience/HKO-7
### IMERG Dataset
It can be obtained via the following address: https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGHH_07/summary

## Start
```
#train
bash scripts/hko7/hko7_RouterFormer_train.sh
bash scripts/imerg/imerg_RouterFormer_train.sh
#test
bash scripts/hko7/hko7_RouterFormer_test.sh
bash scripts/imerg/imerg_RouterFormer_test.sh
```

## Acknowledgments
Our code is based on OpenSTL and PredFormer. We sincerely appreciate for their contributions.

