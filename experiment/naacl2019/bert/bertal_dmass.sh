#!/usr/bin/env bash


#SBATCH --cluster=gpu
#SBATCH --gres=gpu:2
#SBATCH --partition=gtx1080
#SBATCH --nodelist=gpu-n23
#SBATCH --job-name=bertal_dmass
#SBATCH --output=bertal_dmass.out
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
srun python ../../../model/train.py -ngpus 1 -bsize 32 -fw transformer -out bertbaseal_dmass -layer_drop 0.0 -op adagrad -lr 0.01 --mode transbert -nh 4 -nhl 4 -nel 4 -ndl 4 -lc True -eval_freq 0 --fetch_mode tf_example_dataset --subword_vocab_size 0 --dmode alter --tie_embedding all --bert_mode bert_token:bertbase --number_samples 4096 --memory rule -memcfg mofinal --memory_prepare_step 2000000 -warm /zfs1/hdaqing/saz31/text_simplification_0924/bertbaseal/log/model.ckpt-4503976

#python train.py -ngpus 1 -bsize 12 -fw transformer -out bertbase2 -layer_drop 0.0 -op sgd -lr 0.01 --mode transbert -nh 4 -nhl 4 -nel 4 -ndl 4 -lc True -eval_freq 0 --fetch_mode tf_example_dataset --subword_vocab_size 0 --dmode alter --tie_embedding enc_dec --bert_mode bert_token:bertbase &

