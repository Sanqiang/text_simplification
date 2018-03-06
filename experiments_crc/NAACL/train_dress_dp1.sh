#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gtx1080
#SBATCH --job-name=tdr_dp1
#SBATCH --output=tdr_dp1.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# Load modules
module restore

# Run the job
srun python ../model/train.py -fw transformer -out dr_dp1 -layer_drop 0.1 -op adagrad -lr 0.1 --mode dress -warm /ihome/hdaqing/saz31/sanqiang/dr/output/model.ckpt-211000