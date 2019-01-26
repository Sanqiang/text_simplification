#!/usr/bin/env bash


#!/usr/bin/env bash


#SBATCH --cluster=smp
#SBATCH --job-name=spacy_ner
#SBATCH --output=spacy_ner.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --qos=normal
#SBATCH --mem=32g

# Load modules
module restore
export PYTHONPATH="${PYTHONPATH}:/zfs1/hdaqing/saz31/dataset/tmp_trans/code"

# Run the job
srun python ../spacy_ner.py

