#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gtx1080
#SBATCH --job-name=tdr_l2
#SBATCH --output=tdr_l2.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# Load modules
module restore

# Run the job
srun python ../model/train.py -fw transformer -out dr_l2 -layer_drop 0.2 -op adagrad -lr 0.1 --mode dress -nhl 2 -nel 2 -ndl 2 -warm /ihome/hdaqing/saz31/sanqiang/dr/model/model.ckpt-210000
