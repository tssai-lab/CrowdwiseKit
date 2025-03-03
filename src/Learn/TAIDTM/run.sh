#!/bin/bash
#!/usr/bin/python3.8
a=(0.1 0.2 0.3 0.4 0.5)
for j in {0..4}; do
for i in {1..3}; do
{
python3.8 train.py --device 0 --data_root  ./data/fmnist/    --experiment_name fmnist --warmup_epoch 5 --TM_epoch 5 --FT_TM_epoch 2 --GCN_TM_epoch 10 --epoch 60 --warmup_lr 0.01 --TM_lr 1e-2 --FT_TM_lr 1e-2 --lr 1e-2  --expert_num 300 --expert_type_num 3 --th 0.1 --noise_rate ${a[$j]} --k 50 --svd_k 5 --no_grad --seed $i --redun 2
}
done
done
