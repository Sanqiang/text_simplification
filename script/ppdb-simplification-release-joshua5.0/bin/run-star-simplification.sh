#!/bin/bash

#$ -cwd
#$ -V
#$ -pe parallel 8
#$ -l mem=40G
#$ -l h_cpu=86400
#$ -S /bin/bash
#$ -M xwe@cis.upenn.edu
#$ -m eas
#$ -j y -o ./logs/


. ~/.bashrc

set -u

#if [ ! -d /scratch/users/xwe ]; then
#  mkdir /scratch/users/xwe
#fi

date >> ./logs/timing-$JOBID

$JOSHUA/scripts/training/pipeline.pl \
    --rundir pro-star-${1} \
    --readme "Syntax simplification with ${1}" \
    --type hiero \
    --no-prepare \
    --no-corpus-lm \
    --alignment $LOCALDIR/experiments/ppdb-simplification/fake/align \
    --corpus $LOCALDIR/experiments/ppdb-simplification/fake/corpus \
    --tune $LOCALDIR/data/simplification-corpus-turk/tune.8turkers.tok \
    --test $LOCALDIR/data/simplification-corpus-turk/test.8turkers.tok \
    --source norm \
    --target simp \
    --tuner pro \
    --pro-config $LOCALDIR/experiments/ppdb-simplification/pro_star.config \
    --first-step TUNE \
    --tune-grammar $LOCALDIR/experiments/ppdb-simplification/pro-ppdb-1.0-xl-all-simp/data/tune/grammar.packed \
    --test-grammar $LOCALDIR/experiments/ppdb-simplification/pro-ppdb-1.0-xl-all-simp/data/test/grammar.packed \
    --glue-grammar $LOCALDIR/experiments/ppdb-simplification/pro-ppdb-1.0-xl-all-simp/data/tune/grammar.glue \
    --threads 8 \
    --lmfile $LOCALDIR/data/lm/lm-merged.kenlm \
    --joshua-mem 100g \
    --optimizer-runs 2 \
    --tmp $LOCALDIR/experiments/ppdb-simplification/tmp \
    --joshua-config $LOCALDIR/experiments/ppdb-simplification/joshua.config \
    --params-txt $LOCALDIR/experiments/ppdb-simplification/params.txt 

# copy desired output results into $HOME
# delete temporary scratch data in $RUNDIR/pro-${1}

date >> ./logs/timing-$JOBID


#    --grammar $LOCALDIR/data/ppdb/${1}.gz \
