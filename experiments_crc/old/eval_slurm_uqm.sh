#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cluster=gpu
#SBATCH --partition=gtx1080
#SBATCH --job-name=vl_uqm
#SBATCH --output=vl_uqm.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# Load modules
module restore
# module purge
# module load cuda/8.0.44
# module load tensorflow/1.3.0

export PERL5LIB="/ihome/hdaqing/saz31/perl_lib"

# Run the job
srun python model/eval.py --framework transformer --use_quality_model True --output_folder uqm
