#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH -c 8
#SBATCH --gres=gpu:L40:1
#SBATCH --array=1-18

# configs: num_quantizers, codebook_size
num_quantizers_list=(8 8 8 32 32 32 64 64 64)
codebook_size_list=(256 512 1024 256 512 1024 256 512 1024)

config_index=$(($SLURM_ARRAY_TASK_ID - 1))
# if config index > 9, then we need to subtract 9 from it and set learning_rate to 0.01
if [ $config_index -ge 8 ];
then
    config_index=$(($config_index - 8))
    learning_rate=0.01
else
    learning_rate=0.001
fi
num_quantizers=${num_quantizers_list[$config_index]}
codebook_size=${codebook_size_list[$config_index]}


python train_quantizer.py --num_quantizers $num_quantizers --codebook_size $codebook_size --learning_rate $learning_rate