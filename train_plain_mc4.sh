#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gtx1080
#SBATCH --job-name=tr_pl4
#SBATCH --output=tr_pl4.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# Load modules
module restore
# module purge
# module load cuda/8.0.44
# module load tensorflow/1.3.0

# Run the job
srun python model/train.py --framework transformer --output_folder plain4 --tied_embedding enc_dec --optimizer adagrad --min_count 4