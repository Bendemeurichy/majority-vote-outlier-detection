#!/bin/bash -l
#PBS -l walltime=1:00:00
#PBS -l mem=4gb

# $qsub install_venv.sh -v cluster=joltik

module load CUDA-Python/12.1.0-gfbf-2023a-CUDA-12.1.1

PIP_DIR="$VSC_SCRATCH/site-packages"
CACHE_DIR="$VSC_SCRATCH/.cache"

if [ -z $cluster ]; then
	echo "Provide a cluster argument using -v, e.g.: qsub install_venv -v cluster=joltik" 1>&2
	exit 1
fi

VENV_NAME="venv_${cluster}"

if [ -d "$VENV_NAME" ];
then
    rm -rf "$VENV_NAME"
fi

python3 -m venv "$VENV_NAME"
source "$VENV_NAME"/bin/activate
pip install --upgrade -t "$PIP_DIR" --cache-dir="$CACHE_DIR/pip" pandas numpy scikit-learn tifffile tqdm


