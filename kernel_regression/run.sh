#!/bin/bash
#SBATCH --job-name=kernel_reg
#SBATCH --output=./kernel_regression/results/output.log
#SBATCH --error=./kernel_regression/results/error.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=20G
#SBATCH --partition=gpu100


unset SLURM_EXPORT_ENV

export OMP_NUM_THREADS=1


srun /usr/bin/nvidia-smi 
# Activate the conda environment
source /nfs/scistore13/mondegrp/dwu/VENV_IMDB/bin/activate 

# Run the experiment
srun python ./kernel_regression/main.py ./kernel_regression/config/test.yaml --savepath ./kernel_regression/results/test

srun python ./kernel_regression/plots.py ./kernel_regression/results/test  --filename results.pt 

deactivate