#!/bin/bash
#SBATCH --job-name=tr-2.0
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
##SBATCH --gpus-per-node=8
#SBATCH -p superChip -N 1
##SBATCH --gres=gpu:4

# Load the python module first
module load python3

# Change to the directory where this script was submitted
cd "$SLURM_SUBMIT_DIR"

# Create output directory with timestamp
OUTPUT_DIR="outputs/2.2_$(date +'%Y%m%d_%H%M%S')"
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Copy this script to output directory for reference
cp "$0" "$OUTPUT_DIR/training_job.sh"



# Run training and redirect all outputs to the directory
source /SEAS/home/g21775526/torch-py_3.9-aarm64/bin/activate
module load python3

torchrun --standalone  train-ft.py > "$OUTPUT_DIR/py-output.txt" 2>&1







# Move SLURM output files to output directory
mv train.out "$OUTPUT_DIR/" 2>/dev/null || true
mv train.err "$OUTPUT_DIR/" 2>/dev/null || true

# Copy any generated logs or checkpoints
cp -r log*.txt "$OUTPUT_DIR/" 2>/dev/null || true
cp -r *.pt "$OUTPUT_DIR/" 2>/dev/null || true

echo "End time: $(date)"
echo "All outputs saved to: $OUTPUT_DIR"