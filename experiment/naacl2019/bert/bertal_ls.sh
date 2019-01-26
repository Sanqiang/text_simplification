#!/usr/bin/env bash


#SBATCH --cluster=gpu
#SBATCH --gres=gpu:4
#SBATCH --partition=titanx
#SBATCH --job-name=bertal_wkls
#SBATCH --output=bertal_wkls.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6-00:00:00
#SBATCH --qos=long
#SBATCH --mem=64g
#SBATCH --mail-user=saz31@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails


# Load modules
module restore

# Run the job
srun python ../../../model/train.py -ngpus 1 -bsize 10 -fw transformer -out bertal_wkls -layer_drop 0.0 -op adagrad -lr 0.01 --mode transbert -nh 8 -nhl 6 -nel 6 -ndl 6 -lc True -eval_freq 0 --fetch_mode tf_example_dataset --subword_vocab_size 0 --dmode wk --tie_embedding all --bert_mode bert_token:bertbase --number_samples 4096 --train_mode static_seq -warm /zfs1/hdaqing/saz31/text_simplification_0924/bertal_wk/model/model.ckpt-4535134

