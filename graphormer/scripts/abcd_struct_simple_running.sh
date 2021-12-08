#!/usr/bin/env bash

[ -z "${exp_name}" ] && exp_name="abcd-struct"
[ -z "${seed}" ] && seed="1"
[ -z "${arch}" ] && arch="--ffn_dim 80 --hidden_dim 80 --num_heads 8 --dropout_rate 0.1 --n_layers 12 --peak_lr 2e-4 --edge_type multi_hop --multi_hop_max_dist 20"
[ -z "${warmup_updates}" ] && warmup_updates="5000" # 25000
[ -z "${tot_updates}" ] && tot_updates="80000" # 400000
[ -z "${batch_size}" ] && batch_size="32"

echo -e "\n\n"
echo "=====================================ARGS======================================"
echo "arg0: $0"
echo "arch: ${arch}"
echo "seed: ${seed}"
echo "exp_name: ${exp_name}"
echo "warmup_updates: ${warmup_updates}"
echo "tot_updates: ${tot_updates}"
echo "==============================================================================="

save_path_train="../../exps/abcd-struct/$exp_name-$warmup_updates-$tot_updates/$seed"
mkdir -p $save_path_train
#ckp="../../exps/abcd-struct/$exp_name-$warmup_updates-$tot_updates/$seed/lightning_logs/checkpoints"
#CUDA_VISIBLE_DEVICES=2 \
      # train and validation
      python ../../graphormer/entry.py --num_workers 8 --seed $seed --batch_size $batch_size \
      --dataset_name abcd-struct \
      --gpus 2 --accelerator ddp --precision 16 \
      $arch \
      --check_val_every_n_epoch 10 --warmup_updates $warmup_updates --tot_updates $tot_updates \
      --default_root_dir $save_path_train 