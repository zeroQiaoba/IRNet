# IRNet: Iterative Refinement Network for Noisy Partial Label Learning  

Correspondence to: 
  - Zheng Lian (lianzheng2016@ia.ac.cn)
  - Mingyu Xu (xumingyu2021@ia.ac.cn)

## Paper
[**IRNet: Iterative Refinement Network for Noisy Partial Label Learning**](https://arxiv.org/pdf/2211.04774.pdf)<br>
Zheng Lian, Mingyu Xu, Lan Chen, Lei Feng, Bin Liu, Jianhua Tao<br>

Please cite our paper if you find our work useful for your research:

```tex
@article{lian2022arnet,
  title={ARNet: Automatic Refinement Network for Noisy Partial Label Learning},
  author={Lian, Zheng and Xu, Mingyu and Chen, Lan and Feng, Lei and Liu, Bin and Tao, Jianhua},
  journal={arXiv preprint arXiv:2211.04774},
  year={2022}
}
```

## Usage

### Datasets

~~~~shell
# download dataset and put it into ./dataset (or you can download it via torchvision)
https://drive.google.com/file/d/18YrX6JFzOpG2a0OW1jyG65DFgG6r1Seg/view   ->   ./dataset
~~~~



### Run IRNet

~~~~shell
cd irnet
python -u train_merge.py --dataset='cifar10' --partial_rate=0.3  --noise_rate=0.3 --epochs=1000 --encoder='resnet' --lr=0.01 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0 --correct_auto --correct_autowin=100 --correct_threshold_range='0.008,0.008' --correct_type='cluster' --correct_update='case3' --loss_type='SCE' --sce_alpha=6.0  --sce_beta=1.0
~~~~



### Run PiCO Baseline

```shell
cd irnet
python -u train_merge.py --dataset='cifar10' --partial_rate=0.3  --noise_rate=0.3 --epochs=1000 --encoder='resnet' --lr=0.01 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0
```


For other datasets and other settings, please refer to **run.sh**



### Acknowledgement

Thanks to [PiCO](https://github.com/hbzju/PiCO), [RC&CC](https://lfeng-ntu.github.io/Code/RCCC.zip), [PRODEN](https://github.com/Lvcrezia77/PRODEN), [LWC&LWS](https://github.com/hongwei-wen/LW-loss-for-partial-label), and [LOG](https://lfeng-ntu.github.io/Code/LMCL.zip).
