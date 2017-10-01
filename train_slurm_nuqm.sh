#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gtx1080
#SBATCH --job-name=tr_nuqm
#SBATCH --output=tr_nuqm.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# Load modules
module restore
# module purge
# module load cuda/8.0.44
# module load tensorflow/1.3.0

# Run the job
srun python model/train.py --framework transformer --use_quality_model False --output_folder nuqm --warm_start /ihome/hdaqing/saz31/sanqiang/notiming/model/model.ckpt-43000
