#!/usr/bin/env bash


#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gtx1080
#SBATCH --nodelist=gpu-n24
#SBATCH --job-name=dresstkwk
#SBATCH --output=dresstkwk.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6-00:00:00
#SBATCH --qos=long
#SBATCH --mail-user=saz31@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails


# Load modules
module restore

# Run the job
srun python ../../../model/train.py -ngpus 1 -bsize 64 -fw transformer -out dresstkwk -layer_drop 0.2 -op adagrad -lr 0.01 --mode dresstk -dim 300 -nh 4 -nhl 4 -nel 4 -ndl 4 -lc True -eval_freq 0 --subword_vocab_size 0 --fetch_mode tf_example_dataset --dmode wk
#srun python ../../../model/train.py -ngpus 1 -bsize 45 -fw transformer -out dresstk -layer_drop 0.0 -op adagrad -lr 0.01 --mode trans -dim 300 -nh 4 -nhl 4 -nel 4 -ndl 4 -lc True -eval_freq 0 --fetch_mode tf_example_dataset --subword_vocab_size 30000 --dmode alter --tie_embedding enc_dec --pointer_mode fuse --seg_mode cp --warm_start /zfs1/hdaqing/saz31/text_simplification_0924/ckpt/model.ckpt-135322


