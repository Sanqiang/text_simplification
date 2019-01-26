v#!/usr/bin/env bash


#!/usr/bin/env bash


#SBATCH --cluster=mpi
#SBATCH --partition=opa-high-mem
#SBATCH --job-name=struc
#SBATCH --output=struc.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --qos=normal
#SBATCH --mem=64g

# Load modules
module restore

# Run the job
srun python ../structure_features.py

