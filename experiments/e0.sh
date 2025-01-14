#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a6000:1
#SBATCH --job-name=houjun-weberta-e0
#SBATCH --mem=16G
#SBATCH --open-mode=append
#SBATCH --output=./logs/e0.log
#SBATCH --partition=jag-standard
#SBATCH --time=14-0

cd .
source .venv/bin/activate

python main.py roberta_base --wandb

