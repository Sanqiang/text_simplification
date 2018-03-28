#!/usr/bin/env bash


#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=titan
#SBATCH --job-name=wiki_base_sub_eval
#SBATCH --output=wiki_base_sub_eval.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6-00:00:00
#SBATCH --qos=long

# Load modules
module restore

# Run the job
srun python ../model/eval.py -ngpus 1 -bsize 32 -dim 512 -svoc_size 50000 -fw transformer -out wiki_base_sub --mode wiki -nhl 6 -nel 6 -ndl 6 -nh 8 -lc True --it_train True --tied_embedding enc_dec


