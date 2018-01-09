#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gtx1080
#SBATCH --job-name=tdr_l4_ppdb_args
#SBATCH --output=tdr_l4_ppdb_args.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# Load modules
module restore

# Run the job
srun python ../model/test.py -fw transformer -out dr_l4_ppdb_args -layer_drop 0.2 -op adagrad -lr 0.1 --mode dress -nhl 4 -nel 4 -ndl 4 -ppdb 'comp' -lc True -ppdbw 2.0|1.5 --test_ckpt /ihome/hdaqing/saz31/sanqiang/dr_l4_ppdb_args/model/model.ckpt-749572