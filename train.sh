#!/bin/bash
#SBATCH --job-name=train-aladdin
#SBATCH --output=train-aladdin.out
#SBATCH --error=train-aladdin.err
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH -p small-gpu -N 1
# Load the python module first
module load python3

# Change to the directory where this script was submitted
cd "$SLURM_SUBMIT_DIR"

# Create and activate virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Use python and pip from the virtual environment explicitly
if python -m pip install --upgrade pip && \
   python -m pip install torch datasets tiktoken tqdm numpy; then
    echo "Hello, World from $(hostname)! All packages installed successfully."
    echo "xxxxxxxxxxxxxxxxxxxxxxxxxxx=================Training gpt===============================xxxxxxxxxxxxxxxxx"
    echo "Running training on $(hostname) using ${SLURM_GPUS_PER_NODE} GPUs"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    torchrun --standalone --nproc_per_node=1 train_gpt2.py > python_output.txt 2>&1
	#python3 arabic_101B.py
else
    echo "Package installation failed!" >&2
    exit 1
fi

