#!/bin/bash
#SBATCH --job-name=small-gpt
#SBATCH --output=small-gpt%j.out
#SBATCH --error=small-gpt%j.err
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH -p large-gpu -N 1
#SBATCH --gres=gpu:4               # request GPUs (adjust as needed)

module load python3
cd "$SLURM_SUBMIT_DIR"

OUTPUT_DIR="outputs/small-gpt_$(date +'%Y%m%d_%H%M%S')"
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
cp "$0" "$OUTPUT_DIR/training_job.sh"

VENV_DIR="${SLURM_SUBMIT_DIR}/venv_torch"
PYPI_LOG="$OUTPUT_DIR/venv_setup.log"

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtualenv at $VENV_DIR" | tee -a "$PYPI_LOG"
  python3 -m venv "$VENV_DIR" >>"$PYPI_LOG" 2>&1 || { echo "venv creation failed" | tee -a "$PYPI_LOG"; exit 1; }
  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip setuptools wheel >>"$PYPI_LOG" 2>&1
  python -m pip install torch transformers datasets accelerate tokenizers sentencepiece huggingface-hub tqdm numpy einops >>"$PYPI_LOG" 2>&1 || {
    echo "pip install failed - check $PYPI_LOG" | tee -a "$PYPI_LOG"
    deactivate 2>/dev/null || true
    exit 1
  }
  deactivate
else
  echo "Virtualenv already exists at $VENV_DIR" | tee -a "$PYPI_LOG"
fi

source "$VENV_DIR/bin/activate"

# --- determine GPU count at runtime and set torchrun accordingly ---
# Prefer SLURM environment variable if provided, otherwise query nvidia-smi.
if [ -n "$SLURM_GPUS_ON_NODE" ]; then
  GPUS_PER_NODE="$SLURM_GPUS_ON_NODE"
elif [ -n "$SLURM_JOB_GPUS" ]; then
  GPUS_PER_NODE="$SLURM_JOB_GPUS"
else
  GPUS_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l)
  GPUS_PER_NODE=${GPUS_PER_NODE:-1}
fi

# If CUDA_VISIBLE_DEVICES is set by the scheduler, count entries instead
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
  # count comma-separated device ids
  visible_count=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
  GPUS_PER_NODE=$visible_count
fi

echo "Using GPUS_PER_NODE=${GPUS_PER_NODE}" | tee -a "$PYPI_LOG"

# Optional: explicitly restrict visible devices to first N (uncomment if needed)
# export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS_PER_NODE-1)))

# Run training with torchrun using detected GPU count
torchrun --standalone --nproc_per_node="$GPUS_PER_NODE" regular-transformer.py \
    --output_dir "$OUTPUT_DIR" \
    --dataset "data/fineweb2-msa" \
    > "$OUTPUT_DIR/py-output.txt" 2>&1

deactivate 2>/dev/null || true

mv small-gpt_${SLURM_JOB_ID}.out "$OUTPUT_DIR/" 2>/dev/null || true
mv small-gpt_${SLURM_JOB_ID}.err "$OUTPUT_DIR/" 2>/dev/null || true
cp -r log*.txt "$OUTPUT_DIR/" 2>/dev/null || true
cp -r *.pt "$OUTPUT_DIR/" 2>/dev/null || true

echo "End time: $(date)"
echo "All outputs saved to: $OUTPUT_DIR"