#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gtx1080
#SBATCH --job-name=tdr_h256
#SBATCH --output=tdr_h256.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# Load modules
module restore

# Run the job
srun python ../model/train.py -fw transformer -out dr_h256 -layer_drop 0.2 -op adagrad -lr 0.1 --mode dress -dim 256 -nh 4  -nhl 0 -nel 2 -ndl 2
