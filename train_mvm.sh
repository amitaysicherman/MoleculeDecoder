#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH -c 8
#SBATCH --gres=gpu:L40:1
#SBATCH --array=0-26

sizes_list=('m' 'm' 'm' 'm' 'm' 'm' 'm' 'm' 'm' 's' 's' 's' 's' 's' 's' 's' 's' 's' 'l' 'l' 'l' 'l' 'l' 'l' 'l' 'l' 'l')
lr_list=(0.0001 0.0001 0.0001 0.001 0.001 0.001 1e-05 1e-05 1e-05 0.0001 0.0001 0.0001 0.001 0.001 0.001 1e-05 1e-05 1e-05 0.0001 0.0001 0.0001 0.001 0.001 0.001 1e-05 1e-05 1e-05)
alpha_list=(0 1 0.5 0 1 0.5 0 1 0.5 0 1 0.5 0 1 0.5 0 1 0.5 0 1 0.5 0 1 0.5 0 1 0.5)




config_index=$(($SLURM_ARRAY_TASK_ID))
size=${sizes_list[$config_index]}
learning_rate=${lr_list[$config_index]}
alpha=${alpha_list[$config_index]}
echo "size: $size, learning_rate: $learning_rate, config_index: $config_index"
python train_mvm.py --size $size --lr $learning_rate --alpha $alpha