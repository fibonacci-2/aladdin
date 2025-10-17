#!/bin/bash
#SBATCH --job-name=arabic_101B
#SBATCH --output=arabic_101B.out
#SBATCH --error=arabic_101B.err
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH -p 384gb -N 1

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
   python -m pip install datasets tiktoken tqdm numpy; then
    echo "Hello, World from $(hostname)! All packages installed successfully."
    echo "xxxxxxxxxxxxxxxxxxxxxxxxxxx=================Training gpt===============================xxxxxxxxxxxxxxxxx"
    echo "Running training on $(hostname) using ${SLURM_GPUS_PER_NODE} GPUs"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    #torchrun --standalone --nproc_per_node=2 train_gpt2.py > python_output.txt 2>&1
	python3 ar_llmu.py
else
    echo "Package installation failed!" >&2
    exit 1
fi

