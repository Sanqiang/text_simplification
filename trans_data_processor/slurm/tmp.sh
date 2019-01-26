#!/usr/bin/env bash

#SBATCH --cluster=smp
#SBATCH --job-name=prep
#SBATCH --output=prep.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6-00:00:00
#SBATCH --qos=long
#SBATCH --mem=32g

# Load modules
module restore

srun java -cp .:/zfs1/hdaqing/saz31/dataset/tmp_trans/javacode/stanford-eneglish-corenlp-2018-02-27-models.jar:/zfs1/hdaqing/saz31/dataset/tmp_trans/javacode/stanford-ner-3.9.2.jar PreprocessUtil2
