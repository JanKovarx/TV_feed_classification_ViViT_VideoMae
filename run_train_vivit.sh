#!/bin/bash
#PBS -N ViViT_train
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=256gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -e /storage/plzen4-ntis/home/user/file/vivit_error.log
#PBS -o /storage/plzen4-ntis/home/user/file/vivit_output.log

cd /storage/plzen4-ntis/home/user/file

export PYTHONPATH=$PYTHONPATH:$(pwd)
export CONDA_ENVS_PATH=""

# module add mambaforge
# conda activate ""

export WANDB__SERVICE_WAIT=300
export WANDB_API_KEY=""
export WANDB_ENTITY=""

export SINGULARITY_IMAGE="/storage/plzen4-ntis/projects/korpusy_cv/RAVDAI/environment/vivit-05.sif"

singularity run --nv $SINGULARITY_IMAGE python train_vivit.py --config config_vivit.yaml --verbose
