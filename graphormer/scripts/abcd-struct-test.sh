#!/usr/bin/env bash

[ -z "${exp_name}" ] && exp_name="abcd-struct-test"
[ -z "${seed}" ] && seed="1"
[ -z "${arch}" ] && arch="--ffn_dim 80 --hidden_dim 80 --num_heads 8 --dropout_rate 0.1 --n_layers 12 --peak_lr 2e-4 --edge_type multi_hop --multi_hop_max_dist 20"
[ -z "${warmup_updates}" ] && warmup_updates="300" # 25000
[ -z "${tot_updates}" ] && tot_updates="4000" # 400000
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

save_path_train="../../exps/abcd-struct-test/$exp_name-$warmup_updates-$tot_updates/$seed"
checkpoint_dir=$save_path_train/lightning_logs/checkpoints
echo "=====================================EVAL======================================"
for file in `ls $checkpoint_dir/{place your best model}.ckpt`
do
      echo -e "\n\n\n ckpt:"
      echo "$file"
      echo -e "\n\n\n"

      python ../../graphormer/entry.py --num_workers 8 --seed 1 --batch_size $batch_size \
            --dataset_name abcd-struct \
            --gpus 1 --accelerator ddp --precision 16 $arch \
            --default_root_dir tmp/ \
            --checkpoint_path $file --test --progress_bar_refresh_rate 100
done
echo "==============================================================================="