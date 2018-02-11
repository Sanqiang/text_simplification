#!/usr/bin/env bash


#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=titan
#SBATCH --job-name=test_all9
#SBATCH --output=test_all9.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --qos=short

# Load modules
module restore

# Run the job
srun python test_all.py -ngpus 1 -bsize 359 -fw transformer -out dress_final_ffn_rbsing_op_cl2 -layer_drop 0.2 -op adagrad -lr 0.1 --mode dress -nhl 4 -nel 4 -ndl 4 -lc True --memory rule --memory_prepare_step 50000 -memcfg cffn_mofinal_ctxly2 -rbase sing -warm /zfs1/hdaqing/saz31/text_simplification/dress_final_ffn_rbsing_op/model/model.ckpt-249523

