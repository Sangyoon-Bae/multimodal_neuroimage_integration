#!/usr/bin/env bash

[ -z "${exp_name}" ] && exp_name="abcd-struct-valid-rmse-dim4-numhead4-nlayer3"
[ -z "${seed}" ] && seed="1"
[ -z "${arch}" ] && arch="--ffn_dim 4 --hidden_dim 4 --num_heads 4 --intput_dropout_rate 0.0 --attention_dropout_rate 0.1  --dropout_rate 0.1 --n_layers 3 --peak_lr 2e-4 --edge_type multi_hop --multi_hop_max_dist 10"
[ -z "${warmup_updates}" ] && warmup_updates="2500" # 25000
[ -z "${tot_updates}" ] && tot_updates="40000" # 400000
[ -z "${batch_size}" ] && batch_size="64"

echo -e "\n\n"
echo "=====================================ARGS======================================"
echo "arg0: $0"
echo "arch: ${arch}"
echo "seed: ${seed}"
echo "exp_name: ${exp_name}"
echo "warmup_updates: ${warmup_updates}"
echo "tot_updates: ${tot_updates}"
echo "==============================================================================="

default_root_dir="/home/ubuntu/Stella/MLVU_multimodality/Graphormer/exps/abcd-struct/$exp_name-$warmup_updates-$tot_updates/$seed"
mkdir -p $default_root_dir
#ckp="../../exps/abcd-struct/$exp_name-$warmup_updates-$tot_updates/$seed/lightning_logs/checkpoints"
#CUDA_VISIBLE_DEVICES=2 \
      # train and validation
      python ../../graphormer/entry.py --num_workers 8 --seed $seed --batch_size $batch_size \
      --dataset_name abcd-struct \
      --gpus 1 --accelerator ddp --precision 16 \
      $arch \
      --check_val_every_n_epoch 5 --warmup_updates $warmup_updates --tot_updates $tot_updates \
      --default_root_dir $default_root_dir