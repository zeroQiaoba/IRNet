set -e
dataset=$1
loss=$2
partial_rate=$3
noise_rate=$4
model=$5
gpu_id=$6

python -u main.py --dataset=${dataset} --partial_rate=${partial_rate} --noise_rate=${noise_rate} --loss_type=${loss} --lws_weight1=1 --encoder=${model} --lr=1e-2 --weight_decay=1e-2 --epochs=100 --gpu=${gpu_id}
python -u main.py --dataset=${dataset} --partial_rate=${partial_rate} --noise_rate=${noise_rate} --loss_type=${loss} --lws_weight1=1 --encoder=${model} --lr=1e-3 --weight_decay=1e-2 --epochs=100 --gpu=${gpu_id}
python -u main.py --dataset=${dataset} --partial_rate=${partial_rate} --noise_rate=${noise_rate} --loss_type=${loss} --lws_weight1=1 --encoder=${model} --lr=1e-2 --weight_decay=1e-3 --epochs=100 --gpu=${gpu_id}
python -u main.py --dataset=${dataset} --partial_rate=${partial_rate} --noise_rate=${noise_rate} --loss_type=${loss} --lws_weight1=1 --encoder=${model} --lr=1e-3 --weight_decay=1e-3 --epochs=100 --gpu=${gpu_id}

