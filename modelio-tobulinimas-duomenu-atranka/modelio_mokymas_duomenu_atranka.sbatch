#!/bin/bash
#SBATCH --job-name=liepa2_further
#SBATCH --output=logs/liepa2_further_%j.out
#SBATCH --error=logs/liepa2_further_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=80:00:00

set -e

WORKDIR=/scratch/lustre/home/user/workdir
cd $WORKDIR

mkdir -p logs
mkdir -p hf_cache
mkdir -p outputs_further_age_gender

export HF_HOME=$WORKDIR/hf_cache
export HF_DATASETS_CACHE=$WORKDIR/hf_cache

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

singularity exec \
  --bind $WORKDIR \
  --nv \
  /scratch/lustre/home/user/python.sif \
  python3 -u duomenu-atranka.py
