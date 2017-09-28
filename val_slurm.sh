#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cluster=gpu
#SBATCH --partition=gtx1080
#SBATCH --job-name=vl
#SBATCH --output=vl.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# Load modules
module restore
# module purge
# module load cuda/8.0.44
# module load tensorflow/1.3.0

# Copy inputs to scratch
#cp <inputs> $SLURM_SCRATCH
#cd $SLURM_SCRATCH

# Set a trap to copy any temp files you may need
run_on_exit(){
  echo 'error'
  # cp -r $SLURM_SCRATCH/* $SLURM_SUBMIT_DIR
}
trap run_on_exit EXIT 

# Run the job
srun python model/eval.py

# Copy outputs to submit directory
#cp <outputs> $SLURM_SUBMIT_DIR
