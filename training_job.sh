#!/bin/bash
#SBATCH --job-name=tr-2.0
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH -p superChip -N 1
#SBATCH --gres=gpu:4  # Added GPU request

# Load the python module first
module load python3/3.11

# Change to the directory where this script was submitted
cd "$SLURM_SUBMIT_DIR"


source /SEAS/home/g21775526/torch-x86/bin/activate

torchrun --standalone --nproc_per_node=2 train_gpt2.py > python_output.txt 2>&1


