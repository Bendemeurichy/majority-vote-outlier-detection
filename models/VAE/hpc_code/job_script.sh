#!/bin/bash
#PBS -N VAE_Training  
#PBS -l walltime=11:00:00
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l mem=32gb

PIP_DIR="$VSC_SCRATCH/site-packages" # directory to install packages
CACHE_DIR="$VSC_SCRATCH/.cache" # directory to use as cache



source venv_joltik/bin/activate
module load CUDA-Python/12.1.0-gfbf-2023a-CUDA-12.1.1
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

cd $PBS_O_WORKDIR

PYTHONPATH="$PYTHONPATH:$PIP_DIR" python hpc_training.py


