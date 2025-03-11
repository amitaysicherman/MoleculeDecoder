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
    python  autoencoder/train_mvm.py --molformer --train_encoder --train_decoder --batch_size 128 --size sm
elif [ $split_index -eq 1 ]
then
    python  autoencoder/train_mvm.py --molformer --train_encoder --train_decoder --batch_size 128 --size m --dropout 0.2
elif [ $split_index -eq 2 ]
then
    python  autoencoder/train_mvm.py --molformer --train_encoder --train_decoder --batch_size 128 --size m --dropout 0.3
elif [ $split_index -eq 3 ]
then
    python  autoencoder/train_mvm.py --molformer --train_encoder --train_decoder --batch_size 128 --size m --dropout 0.4
elif [ $split_index -eq 4 ]
then
    python  autoencoder/train_mvm.py --molformer --train_encoder --train_decoder --batch_size 128 --size m --dropout 0.0
elif [ $split_index -eq 5 ]
then
    python  autoencoder/train_mvm.py --molformer --train_encoder --train_decoder --batch_size 128 --size sm --dropout 0.0
elif [ $split_index -eq 6 ]
then
    python  autoencoder/train_mvm_retro.py --batch_size 128 --size sm
elif [ $split_index -eq 7 ]
then
    python  autoencoder/train_mvm_retro.py --batch_size 128 --size m --dropout 0.0
fi


