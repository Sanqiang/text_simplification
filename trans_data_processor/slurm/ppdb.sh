#!/usr/bin/env bash


#!/usr/bin/env bash


#SBATCH --cluster=mpi
#SBATCH --partition=opa-high-mem
#SBATCH --job-name=ppdb
#SBATCH --output=ppdb.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --qos=normal
#SBATCH --mem=64g

# Load modules
module restore
module add python/anaconda2.7-5.2.0
export PYTHONPATH="${PYTHONPATH}:/zfs1/hdaqing/saz31/dataset/tmp_trans/code"

# Run the job
srun python ../pppdb_features.py

