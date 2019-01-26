#!/usr/bin/env bash


#SBATCH --cluster=smp
#SBATCH --job-name=vocab
#SBATCH --output=vocab.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6-00:00:00
#SBATCH --qos=long
#SBATCH --mem=32g

# Load modules
module restore
export PYTHONPATH="${PYTHONPATH}:/zfs1/hdaqing/saz31/dataset/tmp_trans/code"
# Run the job
srun python ../vocab_prepare.py