#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH -c 8
#SBATCH --gres=gpu:L40:4
#SBATCH --array=0-1

sizes_list=('l' 'xl')

config_index=$(($SLURM_ARRAY_TASK_ID))
size=${sizes_list[$config_index]}
echo "size: $size config_index: $config_index"
python train_mvm.py --size $size