#!/usr/bin/env bash


#SBATCH --cluster=gpu
#SBATCH --gres=gpu:2
#SBATCH --partition=gtx1080
#SBATCH --nodelist=gpu-n18
#SBATCH --job-name=bertal_comb
#SBATCH --output=bertal_comb.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6-00:00:00
#SBATCH --qos=long
#SBATCH --mem=16g
#SBATCH --mail-user=saz31@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails


# Load modules
module restore

# Run the job
srun python ../../../model/train.py -ngpus 1 -bsize 16 -fw transformer -out bertbaseal_comb -layer_drop 0.0 -op adagrad -lr 0.01 --mode comb -nh 8 -nhl 6 -nel 6 -ndl 6 -lc True -eval_freq 0 --fetch_mode tf_example_dataset --subword_vocab_size 0 --tie_embedding all --bert_mode bert_token:bertbase --number_samples 4096

