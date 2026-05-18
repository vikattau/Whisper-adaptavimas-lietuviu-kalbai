#!/bin/bash
#SBATCH --job-name=liepa2_download
#SBATCH --output=logs/liepa2_%j.out
#SBATCH --error=logs/liepa2_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00

set -e
WORKDIR=/scratch/lustre/home/vita0189/workdir
cd $WORKDIR

mkdir -p logs
mkdir -p cache

echo "Job started on $(hostname)"
echo "Working dir: $(pwd)"
echo "Start time: $(date)"

singularity exec --bind $WORKDIR --nv /scratch/lustre/home/vita0189/python.sif python3 download_liepa2_train.py

echo "End time: $(date)"
echo "Done"