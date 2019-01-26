#!/usr/bin/env bash


#SBATCH --cluster=gpu
#SBATCH --gres=gpu:2
#SBATCH --partition=gtx1080
#SBATCH --nodelist=gpu-n17
#SBATCH --job-name=bertal
#SBATCH --output=bertal.out
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
srun python ../../../model/train.py -ngpus 1 -bsize 20 -fw transformer -out bertbaseal2 -layer_drop 0.0 -op adagrad -lr 0.01 --mode transbert -nh 8 -nhl 6 -nel 6 -ndl 6 -lc True -eval_freq 0 --fetch_mode tf_example_dataset --subword_vocab_size 0 --dmode alter --tie_embedding all --bert_mode bert_token:bertbase --number_samples 4096 -warm /zfs1/hdaqing/saz31/text_simplification_0924/bertal_wk/ckpt/model.ckpt-4002392
#-warm /zfs1/hdaqing/saz31/text_simplification_0924/ckpt/model.ckpt-250442

#python train.py -ngpus 1 -bsize 12 -fw transformer -out bertbase2 -layer_drop 0.0 -op sgd -lr 0.01 --mode transbert -nh 4 -nhl 4 -nel 4 -ndl 4 -lc True -eval_freq 0 --fetch_mode tf_example_dataset --subword_vocab_size 0 --dmode alter --tie_embedding enc_dec --bert_mode bert_token:bertbase &

