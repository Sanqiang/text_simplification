#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gtx1080
#SBATCH --job-name=tr_pl4a
#SBATCH --output=tr_pl4a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# Load modules
module restore
# module purge
# module load cuda/8.0.44
# module load tensorflow/1.3.0

# Run the job
srun python model/train.py --framework transformer --output_folder plain4a --tied_embedding enc_dec --optimizer adagrad --min_count 4 --layer_prepostprocess_dropout 0.2 --learning_rate 0.001 --warm_start /ihome/hdaqing/saz31/sanqiang/plain4/model/model.ckpt-132000 --optimizer adam