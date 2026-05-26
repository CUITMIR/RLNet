# RLNet
## Overview
Overall Architecture of RouterFormer.
<img width="5123" height="2665" alt="图片15" src="https://github.com/user-attachments/assets/354b45d7-7d1c-4259-86fd-c608d27425b5" />


## Installation
```
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
python tools/train.py -d hko7  -c configs/hko7/RLNet.py --ex_name hko7/RLNet -e 50 --batch_size 4 --val_batch_size 4 --gpus 0
#test
python tools/test.py -d hko7  -c configs/hko7/RLNet.py --ex_name test/hko7/RLNet --batch_size 4 --val_batch_size 4 --test --gpus 0 
```

## Acknowledgments
Our code is based on OpenSTL. We sincerely appreciate for their contributions.

