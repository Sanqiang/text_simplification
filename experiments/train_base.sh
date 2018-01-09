#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:3
#SBATCH --partition=gtx1080
#SBATCH --job-name=trans_base
#SBATCH --output=trans_base.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# Load modules
module restore

# Run the job
srun python ../model/train.py -fw transformer -out trans_base -layer_drop 0.2 -op adagrad -lr 0.1 --mode dress -nhl 4 -nel 4 -ndl 4
