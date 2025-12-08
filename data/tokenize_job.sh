#!/bin/bash
#SBATCH --job-name=tokenize
#SBATCH --output=tokenize_%j.out
#SBATCH --error=tokenize_%j.err
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
##SBATCH --gpus-per-node=8
#SBATCH -p highThru -N 1
##SBATCH --gres=gpu:4

# Load the python module first
module load python3

# Change to the directory where this script was submitted
cd "$SLURM_SUBMIT_DIR"



# Run training and redirect all outputs to the directory
source /SEAS/home/g21775526/code/aladdin/.venv/bin/activate
module load python3

python3 aranize_datadet.py
    > "aranize-tweets.out" 2>&1







echo "End time: $(date)"
echo "All outputs saved to:"