#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cluster=gpu
#SBATCH --partition=gtx1080
#SBATCH --job-name=vl_pl4a
#SBATCH --output=vl_pl4a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# Load modules
module restore

export PERL5LIB="/ihome/hdaqing/saz31/perl_lib"

# Run the job
srun python ../model/eval.py -fr transformer -out dr --mode all
