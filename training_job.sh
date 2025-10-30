#!/bin/bash
#SBATCH --job-name=train-gpt
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH --time=165:01:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH -p large-gpu -N 1
#SBATCH --gres=gpu:2  # Added GPU request


# Load the python module first
module load python3

# Change to the directory where this script was submitted
cd "$SLURM_SUBMIT_DIR"

# Create and activate virtual environment if it doesn't exist
if [ ! -d "venv_" ]; then
    python3 -m venv venv_
fi
source venv_/bin/activate

# Use python and pip from the virtual environment explicitly
if python -m pip install --upgrade pip && \
   python -m pip install transformers torch  tiktoken tqdm  numpy; then
    echo "Hello, World from $(hostname)! All packages installed successfully."
    echo "xxxxxxxxxxxxxxxxxxxxxxxxxxx=================Training gpt===============================xxxxxxxxxxxxxxxxx"
    echo "Running training on $(hostname) using ${SLURM_GPUS_PER_NODE} GPUs"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    torchrun --standalone --nproc_per_node=2 train_gpt2.py > python_output.txt 2>&1
else
    echo "Package installation failed!" >&2
    exit 1
fi

