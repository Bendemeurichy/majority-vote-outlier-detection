#!/bin/bash
#PBS -N VAE_Optimization
#PBS -l walltime=6:00:00
#PBS -l nodes=1:ppn=2
#PBS -l mem=64gb

PIP_DIR="$VSC_SCRATCH/site-packages" # directory to install packages
CACHE_DIR="$VSC_SCRATCH/.cache" # directory to use as cache

source venv_joltik/bin/activate
module load PyTorch-bundle/2.1.2-foss-2023a
# pip install --upgrade -t "$PIP_DIR" --cache-dir="$CACHE_DIR/pip" kaleido

cd $PBS_O_WORKDIR

PYTHONPATH="$PYTHONPATH:$PIP_DIR" python hpc_optimization.py