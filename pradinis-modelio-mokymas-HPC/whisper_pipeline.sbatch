#!/bin/bash
#SBATCH --job-name=whisper_pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=80:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

WORKDIR=/scratch/lustre/home/vita0189/workdir
cd $WORKDIR

mkdir -p logs
mkdir -p eval_dataset_cache

singularity exec --nv /scratch/lustre/home/user/python.sif python3 data_setup.py
singularity exec --nv /scratch/lustre/home/user/python.sif python3 checkpoint_optuna.py
singularity exec --nv /scratch/lustre/home/user/python.sif python3 asr_train.py
