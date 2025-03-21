#!/bin/bash

# EXAMPLE USAGE:
# sbatch job.sh configs/default.yaml

#SBATCH --account=mth240012p
#SBATCH --job-name=lab2-autoencoder
#SBATCH --cpus-per-task=5

#SBATCH --time 10:00:00
#SBATCH -o test.out          # write job console output to file test.out
#SBATCH -e test.err          # write job console errors to file test.err

#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1

module load anaconda3
conda activate env_214
python run_autoencoder.py $1