#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:k80:1
#SBATCH --time=2-00:00:00
#SBATCH --qos=maxgpu
#SBATCH --job-name=wiki_base_2
#SBATCH --output=wiki_base_2.out

module restore

srun python ../model/train.py -ngpus 1 -bsize 32 -fw transformer -out wiki_base_add -layer_drop 0.2 -op adagrad -lr 0.1 --mode wiki -nhl 6 -nel 6 -ndl 6 -lc True --it_train True -eval_freq 100000 --use_dataset2 True

