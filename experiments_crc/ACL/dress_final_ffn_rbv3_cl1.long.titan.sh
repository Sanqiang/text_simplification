#!/usr/bin/env bash


#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=titan
#SBATCH --job-name=dress_final_ffn_rbv3_op_cl1
#SBATCH --output=dress_final_ffn_rbv3_op_cl1.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6-00:00:00
#SBATCH --qos=long


# Load modules
module restore

# Run the job
srun python ../model/train.py -ngpus 1 -bsize 64 -fw transformer -out dress_final_ffn_rbv3_op_cl1 -layer_drop 0.2 -op adagrad -lr 0.1 --mode dress -nhl 4 -nel 4 -ndl 4 -lc True --memory rule --memory_prepare_step 50000 -memcfg cffn_mofinal_ctxly1 -rbase v3 -warm /zfs1/hdaqing/saz31/text_simplification/dress_final_ffn_rbv3_op/model/model.ckpt-609491

