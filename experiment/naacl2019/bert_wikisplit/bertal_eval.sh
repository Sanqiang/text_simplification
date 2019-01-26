#!/usr/bin/env bash


#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=titan
#SBATCH --job-name=bertal_ws_eval
#SBATCH --output=bertal_ws_eval.out
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
srun python ../../../model/eval.py -ngpus 1 -bsize 12 -fw transformer -out bertbaseal_ws2 -layer_drop 0.0 -op adagrad -lr 0.01 --mode wikisplit -nh 8 -nhl 6 -nel 6 -ndl 6 -lc True -eval_freq 0 --fetch_mode tf_example_dataset --subword_vocab_size 0 --tie_embedding all --bert_mode bert_token:bertbase --number_samples 4096

