#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH -c 8
#SBATCH --gres=gpu:L40:1
#SBATCH --array=1-10


split_index=$(($SLURM_ARRAY_TASK_ID - 1))
MILION=1000000
TEN_MILION=10000000
START_INDEX=$(($split_index * $TEN_MILION))
END_INDEX=$(($START_INDEX + $TEN_MILION))

python save_molformer_pubchem_all.py --start_index $START_INDEX --end_index $END_INDEX