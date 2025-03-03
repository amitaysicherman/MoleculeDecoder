#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH -c 8
#SBATCH --gres=gpu:L40:1
#SBATCH --array=0-8


split_index=$(($SLURM_ARRAY_TASK_ID - 1))
#models: ae vae vq
#sizes: s m l
models_lines=("ae" "vae" "vq" "ae" "vae" "vq" "ae" "vae" "vq")
sizes_lines=("s" "s" "s" "m" "m" "m" "l" "l" "l")
model=${models_lines[$split_index]}
size=${sizes_lines[$split_index]}

python autoencoder/main.py --model $model --size $size