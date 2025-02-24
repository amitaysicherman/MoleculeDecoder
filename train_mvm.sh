#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH -c 8
#SBATCH --gres=gpu:L40:1
#SBATCH --array=0-14

sizes_list=("s" "s" "s" "m" "m" "m" "l" "l" "l" "xs" "xs" "xs" "xl" "xl" "xl")
lr_list=(1e-3 1e-4 1e-5 1e-3 1e-4 1e-5 1e-3 1e-4 1e-5 1e-3 1e-4 1e-5 1e-3 1e-4 1e-5)

config_index=$(($SLURM_ARRAY_TASK_ID))
size=${sizes_list[$config_index]}
learning_rate=${lr_list[$config_index]}
echo "size: $size, learning_rate: $learning_rate, config_index: $config_index"
python train_mvm.py --size $size --lr $learning_rate