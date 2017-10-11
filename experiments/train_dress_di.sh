#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gtx1080
#SBATCH --job-name=tdr_dig
#SBATCH --output=tdr_dig.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# Load modules
module restore

# Run the job
srun python ../model/train.py -fw transformer -out dr_dig -layer_drop 0.2 -op adagrad -lr 0.15 --mode dress -digate True -warm /ihome/hdaqing/saz31/sanqiang/dr/model/model.ckpt-152000