#!/bin/bash
#SBATCH --job-name=push_ckpt_3500
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

WORKDIR=/scratch/lustre/home/vita0189/workdir
cd $WORKDIR

mkdir -p logs
mkdir -p outputs

export HF_TOKEN="HF_token"

singularity exec --nv /scratch/lustre/home/vita0189/python.sif python3 from_checkpoint.py

