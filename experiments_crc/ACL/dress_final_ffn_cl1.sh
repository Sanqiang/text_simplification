#!/usr/bin/env bash
partition=$1
qos=$2
name=$3

if [[ -z $3 ]]; then
    name='dress_final_ffn_cl1'
fi

if [ $1 = "g" ]; then
    partition='gtx1080'
elif [ $1 = "tx" ]; then
    partition='titanx'
elif [ $1 = "t" ]; then
    partition='titan'
fi


if [ $2 = "s" ]; then
    time='1-00:00:00'
    qos='short'
elif [ $2 = "n" ]; then
    time='3-00:00:00'
    qos='normal'
elif [ $2 = "l" ]; then
    time='6-00:00:00'
    qos='long'
fi

#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=$partition
#SBATCH --job-name=$name
#SBATCH --output=$name'.out'
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=$time
#SBATCH --qos=$qos

echo 'Task:' $name
echo 'Use Parititon:' $partition
echo 'WIth Time Length:' $time

# Load modules
module restore

# Run the job
srun python ../model/train.py -ngpus 1 -bsize 64 -fw transformer -out $name -layer_drop 0.2 -op adagrad -lr 0.1 --mode dress -nhl 4 -nel 4 -ndl 4 -lc True --memory rule --memory_prepare_step 50000 -memcfg cffn_mofinal_ctxly1


