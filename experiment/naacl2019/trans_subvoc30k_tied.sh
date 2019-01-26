#!/usr/bin/env bash


#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gtx1080
#SBATCH --nodelist=gpu-n24
#SBATCH --job-name=subtok_tied
#SBATCH --output=subtok_tied.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --qos=long
#SBATCH --mem=16g
#SBATCH --mail-user=saz31@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails


# Load modules
module restore

# Run the job
srun python ../../model/train.py -ngpus 1 -bsize 45 -fw transformer -out subtok_tied -layer_drop 0.0 -op adagrad -lr 0.01 --mode trans -dim 300 -nh 4 -nhl 4 -nel 4 -ndl 4 -lc True -eval_freq 0 --fetch_mode tf_example_dataset --subword_vocab_size 30000 --dmode alter --tie_embedding enc_dec


