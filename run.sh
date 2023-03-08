###########################################################################
## CIFAR-10
###########################################################################
=> Baseline: PiCO
cd ./irnet
python -u train_merge.py --dataset='cifar10' --partial_rate=0.1  --noise_rate=0.1 --epochs=1000 --encoder='resnet' --lr=0.01 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0
python -u train_merge.py --dataset='cifar10' --partial_rate=0.1  --noise_rate=0.2 --epochs=1000 --encoder='resnet' --lr=0.01 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0
python -u train_merge.py --dataset='cifar10' --partial_rate=0.1  --noise_rate=0.3 --epochs=1000 --encoder='resnet' --lr=0.01 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0
python -u train_merge.py --dataset='cifar10' --partial_rate=0.3  --noise_rate=0.1 --epochs=1000 --encoder='resnet' --lr=0.01 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0
python -u train_merge.py --dataset='cifar10' --partial_rate=0.3  --noise_rate=0.2 --epochs=1000 --encoder='resnet' --lr=0.01 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0
python -u train_merge.py --dataset='cifar10' --partial_rate=0.3  --noise_rate=0.3 --epochs=1000 --encoder='resnet' --lr=0.01 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0
python -u train_merge.py --dataset='cifar10' --partial_rate=0.5  --noise_rate=0.1 --epochs=1000 --encoder='resnet' --lr=0.01 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0
python -u train_merge.py --dataset='cifar10' --partial_rate=0.5  --noise_rate=0.2 --epochs=1000 --encoder='resnet' --lr=0.01 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0
python -u train_merge.py --dataset='cifar10' --partial_rate=0.5  --noise_rate=0.3 --epochs=1000 --encoder='resnet' --lr=0.01 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0

=> Baseline: rc/cc/log/lwc/lws 
=> [partial_rate] choose from [0.1, 0.3, 0.5], [noise_rate] choose from [0.1, 0.2, 0.3]
cd ./pll-baseline
sh run_rc_cc_exp_log.sh cifar10 rc  [partial_rate] [noise_rate] convnet 0
sh run_rc_cc_exp_log.sh cifar10 cc  [partial_rate] [noise_rate] convnet 0
sh run_rc_cc_exp_log.sh cifar10 log [partial_rate] [noise_rate] convnet 0
sh run_lwc_lws.sh       cifar10 lwc [partial_rate] [noise_rate] convnet 0
sh run_lwc_lws.sh       cifar10 lws [partial_rate] [noise_rate] convnet 0

=> IRNet
cd ./irnet
python -u train_merge.py --dataset='cifar10' --partial_rate=0.1  --noise_rate=0.1 --epochs=1000 --encoder='resnet' --lr=0.01 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0 --correct_auto --correct_autowin=100 --correct_threshold_range='0.020,0.020' --correct_type='cluster' --correct_update='case3' --loss_type='SCE' --sce_alpha=6.0  --sce_beta=1.0
python -u train_merge.py --dataset='cifar10' --partial_rate=0.1  --noise_rate=0.2 --epochs=1000 --encoder='resnet' --lr=0.01 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0 --correct_auto --correct_autowin=100 --correct_threshold_range='0.016,0.016' --correct_type='cluster' --correct_update='case3' --loss_type='SCE' --sce_alpha=6.0  --sce_beta=1.0
python -u train_merge.py --dataset='cifar10' --partial_rate=0.1  --noise_rate=0.3 --epochs=1000 --encoder='resnet' --lr=0.01 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0 --correct_auto --correct_autowin=100 --correct_threshold_range='0.010,0.010' --correct_type='cluster' --correct_update='case3' --loss_type='SCE' --sce_alpha=6.0  --sce_beta=1.0
python -u train_merge.py --dataset='cifar10' --partial_rate=0.3  --noise_rate=0.1 --epochs=1000 --encoder='resnet' --lr=0.01 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0 --correct_auto --correct_autowin=100 --correct_threshold_range='0.016,0.016' --correct_type='cluster' --correct_update='case3' --loss_type='SCE' --sce_alpha=6.0  --sce_beta=1.0
python -u train_merge.py --dataset='cifar10' --partial_rate=0.3  --noise_rate=0.2 --epochs=1000 --encoder='resnet' --lr=0.01 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0 --correct_auto --correct_autowin=100 --correct_threshold_range='0.010,0.010' --correct_type='cluster' --correct_update='case3' --loss_type='SCE' --sce_alpha=6.0  --sce_beta=1.0
python -u train_merge.py --dataset='cifar10' --partial_rate=0.3  --noise_rate=0.3 --epochs=1000 --encoder='resnet' --lr=0.01 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0 --correct_auto --correct_autowin=100 --correct_threshold_range='0.008,0.008' --correct_type='cluster' --correct_update='case3' --loss_type='SCE' --sce_alpha=6.0  --sce_beta=1.0
python -u train_merge.py --dataset='cifar10' --partial_rate=0.5  --noise_rate=0.1 --epochs=1000 --encoder='resnet' --lr=0.01 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0 --correct_auto --correct_autowin=100 --correct_threshold_range='0.010,0.010' --correct_type='cluster' --correct_update='case3' --loss_type='SCE' --sce_alpha=6.0  --sce_beta=1.0
python -u train_merge.py --dataset='cifar10' --partial_rate=0.5  --noise_rate=0.2 --epochs=1000 --encoder='resnet' --lr=0.01 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0 --correct_auto --correct_autowin=100 --correct_threshold_range='0.008,0.008' --correct_type='cluster' --correct_update='case3' --loss_type='SCE' --sce_alpha=6.0  --sce_beta=1.0
python -u train_merge.py --dataset='cifar10' --partial_rate=0.5  --noise_rate=0.3 --epochs=1000 --encoder='resnet' --lr=0.01 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0 --correct_auto --correct_autowin=100 --correct_threshold_range='0.008,0.008' --correct_type='cluster' --correct_update='case3' --loss_type='SCE' --sce_alpha=6.0  --sce_beta=1.0

###########################################################################
## CIFAR-100
###########################################################################
=> Baseline: PiCO
cd ./irnet
python -u train_merge.py --dataset='cifar100' --partial_rate=0.01 --noise_rate=0.1 --epochs=1000 --proto_start=1 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0
python -u train_merge.py --dataset='cifar100' --partial_rate=0.01 --noise_rate=0.2 --epochs=1000 --proto_start=1 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0
python -u train_merge.py --dataset='cifar100' --partial_rate=0.01 --noise_rate=0.3 --epochs=1000 --proto_start=1 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0
python -u train_merge.py --dataset='cifar100' --partial_rate=0.03 --noise_rate=0.1 --epochs=1000 --proto_start=1 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0
python -u train_merge.py --dataset='cifar100' --partial_rate=0.03 --noise_rate=0.2 --epochs=1000 --proto_start=1 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0
python -u train_merge.py --dataset='cifar100' --partial_rate=0.03 --noise_rate=0.3 --epochs=1000 --proto_start=1 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0
python -u train_merge.py --dataset='cifar100' --partial_rate=0.05 --noise_rate=0.1 --epochs=1000 --proto_start=1 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0
python -u train_merge.py --dataset='cifar100' --partial_rate=0.05 --noise_rate=0.2 --epochs=1000 --proto_start=1 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0
python -u train_merge.py --dataset='cifar100' --partial_rate=0.05 --noise_rate=0.3 --epochs=1000 --proto_start=1 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0

=> Baseline: rc/cc/log/lwc/lws
=> [partial_rate] choose from [0.01, 0.03, 0.05], [noise_rate] choose from [0.1, 0.2, 0.3]
cd ./pll-baseline
sh run_rc_cc_exp_log.sh cifar100 rc  [partial_rate] [noise_rate] convnet 0
sh run_rc_cc_exp_log.sh cifar100 cc  [partial_rate] [noise_rate] convnet 0
sh run_rc_cc_exp_log.sh cifar100 log [partial_rate] [noise_rate] convnet 0
sh run_lwc_lws.sh       cifar100 lwc [partial_rate] [noise_rate] convnet 0
sh run_lwc_lws.sh       cifar100 lws [partial_rate] [noise_rate] convnet 0

=> IRNet
cd ./irnet
python -u train_merge.py --dataset='cifar100' --partial_rate=0.01 --noise_rate=0.1 --epochs=1000 --proto_start=1 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0 --correct_auto --correct_autowin=100 --correct_threshold_range='0.0040,0.0040' --correct_type='cluster' --correct_update='case3' --loss_type='CE'
python -u train_merge.py --dataset='cifar100' --partial_rate=0.01 --noise_rate=0.2 --epochs=1000 --proto_start=1 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0 --correct_auto --correct_autowin=100 --correct_threshold_range='0.0020,0.0020' --correct_type='cluster' --correct_update='case3' --loss_type='CE'
python -u train_merge.py --dataset='cifar100' --partial_rate=0.01 --noise_rate=0.3 --epochs=1000 --proto_start=1 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0 --correct_auto --correct_autowin=100 --correct_threshold_range='0.0020,0.0020' --correct_type='cluster' --correct_update='case3' --loss_type='CE'
python -u train_merge.py --dataset='cifar100' --partial_rate=0.03 --noise_rate=0.1 --epochs=1000 --proto_start=1 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0 --correct_auto --correct_autowin=100 --correct_threshold_range='0.0020,0.0020' --correct_type='cluster' --correct_update='case3' --loss_type='CE'
python -u train_merge.py --dataset='cifar100' --partial_rate=0.03 --noise_rate=0.2 --epochs=1000 --proto_start=1 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0 --correct_auto --correct_autowin=100 --correct_threshold_range='0.0020,0.0020' --correct_type='cluster' --correct_update='case3' --loss_type='CE'
python -u train_merge.py --dataset='cifar100' --partial_rate=0.03 --noise_rate=0.3 --epochs=1000 --proto_start=1 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0 --correct_auto --correct_autowin=100 --correct_threshold_range='0.0020,0.0020' --correct_type='cluster' --correct_update='case3' --loss_type='CE'
python -u train_merge.py --dataset='cifar100' --partial_rate=0.05 --noise_rate=0.1 --epochs=1000 --proto_start=1 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0 --correct_auto --correct_autowin=100 --correct_threshold_range='0.0020,0.0020' --correct_type='cluster' --correct_update='case3' --loss_type='CE'
python -u train_merge.py --dataset='cifar100' --partial_rate=0.05 --noise_rate=0.2 --epochs=1000 --proto_start=1 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0 --correct_auto --correct_autowin=100 --correct_threshold_range='0.0020,0.0020' --correct_type='cluster' --correct_update='case3' --loss_type='CE'
python -u train_merge.py --dataset='cifar100' --partial_rate=0.05 --noise_rate=0.3 --epochs=1000 --proto_start=1 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0 --correct_auto --correct_autowin=100 --correct_threshold_range='0.0020,0.0020' --correct_type='cluster' --correct_update='case3' --loss_type='CE'

###########################################################################
## KMNIST
###########################################################################
=> Baseline: PiCO
cd ./irnet
python -u train_merge.py --dataset='kmnist' --partial_rate=0.1 --noise_rate=0.1 --epochs=1000 --proto_start=100 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0
python -u train_merge.py --dataset='kmnist' --partial_rate=0.1 --noise_rate=0.2 --epochs=1000 --proto_start=100 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0
python -u train_merge.py --dataset='kmnist' --partial_rate=0.1 --noise_rate=0.3 --epochs=1000 --proto_start=100 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0
python -u train_merge.py --dataset='kmnist' --partial_rate=0.3 --noise_rate=0.1 --epochs=1000 --proto_start=100 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0
python -u train_merge.py --dataset='kmnist' --partial_rate=0.3 --noise_rate=0.2 --epochs=1000 --proto_start=100 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0
python -u train_merge.py --dataset='kmnist' --partial_rate=0.3 --noise_rate=0.3 --epochs=1000 --proto_start=100 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0
python -u train_merge.py --dataset='kmnist' --partial_rate=0.5 --noise_rate=0.1 --epochs=1000 --proto_start=100 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0
python -u train_merge.py --dataset='kmnist' --partial_rate=0.5 --noise_rate=0.2 --epochs=1000 --proto_start=100 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0
python -u train_merge.py --dataset='kmnist' --partial_rate=0.5 --noise_rate=0.3 --epochs=1000 --proto_start=100 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0

=> Baseline: rc/cc/log/lwc/lws 
=> [partial_rate] choose from [0.1, 0.3, 0.5], [noise_rate] choose from [0.1, 0.2, 0.3]
cd ./pll-baseline
sh run_rc_cc_exp_log.sh kmnist rc  [partial_rate] [noise_rate] mlp 0 
sh run_rc_cc_exp_log.sh kmnist cc  [partial_rate] [noise_rate] mlp 0 
sh run_rc_cc_exp_log.sh kmnist log [partial_rate] [noise_rate] mlp 0 
sh run_lwc_lws.sh       kmnist lwc [partial_rate] [noise_rate] mlp 0 
sh run_lwc_lws.sh       kmnist lws [partial_rate] [noise_rate] mlp 0 

=> IRNet
cd ./irnet
python -u train_merge.py --dataset='kmnist' --partial_rate=0.1 --noise_rate=0.1 --epochs=1000 --proto_start=100 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0 --correct_auto --correct_autowin=100 --correct_threshold_range='0.010,0.010' --correct_type='cluster' --correct_update='case3' --loss_type='SCE' --sce_alpha=6.0 --sce_beta=1.0
python -u train_merge.py --dataset='kmnist' --partial_rate=0.1 --noise_rate=0.2 --epochs=1000 --proto_start=100 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0 --correct_auto --correct_autowin=100 --correct_threshold_range='0.010,0.010' --correct_type='cluster' --correct_update='case3' --loss_type='SCE' --sce_alpha=6.0 --sce_beta=1.0
python -u train_merge.py --dataset='kmnist' --partial_rate=0.1 --noise_rate=0.3 --epochs=1000 --proto_start=100 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0 --correct_auto --correct_autowin=100 --correct_threshold_range='0.010,0.010' --correct_type='cluster' --correct_update='case3' --loss_type='SCE' --sce_alpha=6.0 --sce_beta=1.0
python -u train_merge.py --dataset='kmnist' --partial_rate=0.3 --noise_rate=0.1 --epochs=1000 --proto_start=100 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0 --correct_auto --correct_autowin=100 --correct_threshold_range='0.010,0.010' --correct_type='cluster' --correct_update='case3' --loss_type='SCE' --sce_alpha=6.0 --sce_beta=1.0
python -u train_merge.py --dataset='kmnist' --partial_rate=0.3 --noise_rate=0.2 --epochs=1000 --proto_start=100 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0 --correct_auto --correct_autowin=100 --correct_threshold_range='0.010,0.010' --correct_type='cluster' --correct_update='case3' --loss_type='SCE' --sce_alpha=6.0 --sce_beta=1.0
python -u train_merge.py --dataset='kmnist' --partial_rate=0.3 --noise_rate=0.3 --epochs=1000 --proto_start=100 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0 --correct_auto --correct_autowin=100 --correct_threshold_range='0.010,0.010' --correct_type='cluster' --correct_update='case3' --loss_type='SCE' --sce_alpha=6.0 --sce_beta=1.0
python -u train_merge.py --dataset='kmnist' --partial_rate=0.5 --noise_rate=0.1 --epochs=1000 --proto_start=100 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0 --correct_auto --correct_autowin=100 --correct_threshold_range='0.010,0.010' --correct_type='cluster' --correct_update='case3' --loss_type='SCE' --sce_alpha=6.0 --sce_beta=1.0
python -u train_merge.py --dataset='kmnist' --partial_rate=0.5 --noise_rate=0.2 --epochs=1000 --proto_start=100 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0 --correct_auto --correct_autowin=100 --correct_threshold_range='0.010,0.010' --correct_type='cluster' --correct_update='case3' --loss_type='SCE' --sce_alpha=6.0 --sce_beta=1.0
python -u train_merge.py --dataset='kmnist' --partial_rate=0.5 --noise_rate=0.3 --epochs=1000 --proto_start=100 --encoder='resnet' --lr=1e-2 --lr_adjust='Case1' --optimizer='sgd' --weight_decay=1e-3 --gpu=0 --correct_auto --correct_autowin=100 --correct_threshold_range='0.008,0.008' --correct_type='cluster' --correct_update='case3' --loss_type='SCE' --sce_alpha=6.0 --sce_beta=1.0
