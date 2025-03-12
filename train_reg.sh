#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH -c 8
#SBATCH --gres=gpu:L40:1
#SBATCH --array=0-3


split_index=$(($SLURM_ARRAY_TASK_ID))
export PYTHONPATH=$PYTHONPATH:$(pwd)
if [ $split_index -eq 0 ]
then
    python  python train_regular.py
elif [ $split_index -eq 1 ]
then
    python  python train_regular.py  --retro
elif [ $split_index -eq 2 ]
then
    python  python train_regular.py --parouts 1
elif [ $split_index -eq 3 ]
then
    python  python train_regular.py  --retro --parouts 1
fi



