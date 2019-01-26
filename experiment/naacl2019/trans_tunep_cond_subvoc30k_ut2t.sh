#!/usr/bin/env bash


#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gtx1080
#SBATCH --nodelist=gpu-n20
#SBATCH --job-name=tunep_subtok_ut2t
#SBATCH --output=tunep_subtok_ut2t.out
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
srun python ../../model/train.py -ngpus 1 -bsize 32 -fw transformer -out tunep_subtok_ut2t -layer_drop 0.0 -op adagrad -lr 0.01 --mode trans -dim 300 -nh 4 -nhl 4 -nel 4 -ndl 4 -lc True -eval_freq 0 --fetch_mode tf_example_dataset --tune_style 10.0:0.0:0.0:0.0 --tune_mode plus:decoder:cond --subword_vocab_size 30000 --dmode alter --architecture ut2t
