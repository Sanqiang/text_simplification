#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gtx1080
#SBATCH --job-name=tdr_l4_ppdb_rlks
#SBATCH --output=tdr_l4_ppdb_rlks.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# Load modules
module restore

# Run the job
srun python ../model/train.py -fw transformer -out dr_l4_ppdb_rlks -layer_drop 0.2 -op adagrad -lr 0.1 --mode dress -nhl 4 -nel 4 -ndl 4 -ppdb 'comp' -lc True -rl_keep 1.5 -rl_simp 2.0