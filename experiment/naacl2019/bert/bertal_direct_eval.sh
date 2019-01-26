#!/usr/bin/env bash


#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=titan
#SBATCH --job-name=bertal_direct_eval
#SBATCH --output=bertal_direct_eval.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6-00:00:00
#SBATCH --qos=long
#SBATCH --mem=48g
#SBATCH --mail-user=saz31@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails


# Load modules
module restore

# Run the job
srun python ../../../model/eval.py -ngpus 1 -bsize 16 -fw transformer -out bertbaseal_direct -layer_drop 0.0 -op adagrad -lr 0.01 --mode transbert -nh 4 -nhl 4 -nel 4 -ndl 4 -lc True -eval_freq 0 --subword_vocab_size 0 --dmode alter --tie_embedding all --bert_mode bert_token:bertbase --number_samples 4096 --memory direct



#python train.py -ngpus 1 -bsize 12 -fw transformer -out bertbase2 -layer_drop 0.0 -op sgd -lr 0.01 --mode transbert -nh 4 -nhl 4 -nel 4 -ndl 4 -lc True -eval_freq 0 --fetch_mode tf_example_dataset --subword_vocab_size 0 --dmode alter --tie_embedding enc_dec --bert_mode bert_token:bertbase &

