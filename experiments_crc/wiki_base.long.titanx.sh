#!/usr/bin/env bash


#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=titanx
#SBATCH --job-name=wiki_base_eval
#SBATCH --output=wiki_base_eval.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6-00:00:00
#SBATCH --qos=long

# Load modules
module restore

# Run the job
srun python ../model/train.py -ngpus 1 -bsize 115 -fw transformer -out wiki_base -layer_drop 0.2 -op adagrad -lr 0.1 --mode wiki -nhl 6 -nel 6 -ndl 6 -lc True --it_train True -eval_freq 0 -warm /zfs1/hdaqing/saz31/text_simplification/wiki_base_backup/model/model.ckpt-891866
