#!/usr/bin/env bash

export PYTHONPATH="/home/zhaos5/projs/ts/text_simplification"

#ls
CUDA_VISIBLE_DEVICES=2,4,5,7 nohup python ../../../model/train.py -ngpus 1 -bsize 10 -fw transformer -out bertal_wkls -layer_drop 0.0 -op adagrad -lr 0.01 --mode transbert -nh 8 -nhl 6 -nel 6 -ndl 6 -lc True -eval_freq 0 --fetch_mode tf_example_dataset --subword_vocab_size 0 --dmode wk --tie_embedding all --bert_mode bert_token:bertbase --number_samples 4096 --environment aws --train_mode static_seq -warm /home/zhaos5/projs/ts/ckpt/model.ckpt-4535134 > ls_train.log &
CUDA_VISIBLE_DEVICES=7 nohup python ../../../model/eval.py -ngpus 1 -bsize 10 -fw transformer -out bertal_wkls -layer_drop 0.0 -op adagrad -lr 0.01 --mode transbert -nh 8 -nhl 6 -nel 6 -ndl 6 -lc True -eval_freq 0 --subword_vocab_size 0 --dmode wk --tie_embedding all --bert_mode bert_token:bertbase:init --number_samples 4096 --environment aws > ls_eval.log &

# ori
CUDA_VISIBLE_DEVICES=4,5 nohup python ../../../model/train.py -ngpus 1 -bsize 100 -fw transformer -out bertal_wkori -layer_drop 0.0 -op adagrad -lr 0.01 --mode transbert_ori -nh 8 -nhl 6 -nel 6 -ndl 6 -lc True -eval_freq 0 --fetch_mode tf_example_dataset --subword_vocab_size 0 --dmode wk --tie_embedding all --bert_mode bert_token:bertbase:init --environment aws -warm /home/zhaos5/projs/ts/ckpt/model.ckpt-4535134 > ori_train.log &
CUDA_VISIBLE_DEVICES=7 nohup python ../../../model/eval.py  -ngpus 1 -bsize 50 -fw transformer -out bertal_wkori -layer_drop 0.0 -op adagrad -lr 0.01 --mode transbert_ori -nh 8 -nhl 6 -nel 6 -ndl 6 -lc True -eval_freq 0 --subword_vocab_size 0 --dmode wk --tie_embedding all --bert_mode bert_token:bertbase:init --environment aws > ori_eval.log &
# ori npad
CUDA_VISIBLE_DEVICES=4 nohup python ../../../model/train.py -ngpus 1 -bsize 100 -fw transformer -out bertal_wkori_npad -layer_drop 0.0 -op adagrad -lr 0.01 --mode transbert_ori -nh 8 -nhl 6 -nel 6 -ndl 6 -lc True -eval_freq 0 --fetch_mode tf_example_dataset --subword_vocab_size 0 --dmode wk --tie_embedding all --bert_mode bert_token:bertbase:init --environment aws --npad_mode v1 -warm /home/zhaos5/projs/ts/perf/bertal_wkori/model/model.ckpt-16846 > ori_train_npad.log &
CUDA_VISIBLE_DEVICES=7 nohup python ../../../model/eval.py  -ngpus 1 -bsize 50 -fw transformer -out bertal_wkori_npad -layer_drop 0.0 -op adagrad -lr 0.01 --mode transbert_ori -nh 8 -nhl 6 -nel 6 -ndl 6 -lc True -eval_freq 0 --subword_vocab_size 0 --dmode wk --tie_embedding all --bert_mode bert_token:bertbase:init --environment aws --npad_mode v1 > ori_eval_npad.log &

#ori direct
CUDA_VISIBLE_DEVICES=5,6 nohup python ../../../model/train.py -ngpus 1 -bsize 48 -fw transformer -out bertal_wkori_direct -layer_drop 0.0 -op adagrad -lr 0.01 --mode transbert_ori -nh 8 -nhl 6 -nel 6 -ndl 6 -lc True -eval_freq 0 --fetch_mode tf_example_dataset --subword_vocab_size 0 --dmode wk --tie_embedding all --bert_mode bert_token:bertbase:init --environment aws -warm /home/zhaos5/projs/ts/ckpt/model.ckpt-4535134 --memory direct > ori_train_direct.log &
CUDA_VISIBLE_DEVICES=7 nohup python ../../../model/eval.py  -ngpus 1 -bsize 8 -fw transformer -out bertal_wkori_direct -layer_drop 0.0 -op adagrad -lr 0.01 --mode transbert_ori -nh 8 -nhl 6 -nel 6 -ndl 6 -lc True -eval_freq 0 --subword_vocab_size 0 --dmode wk --tie_embedding all --bert_mode bert_token:bertbase:init --environment aws > ori_eval_direct.log &
#ori direct gate
CUDA_VISIBLE_DEVICES=3,4 nohup python ../../../model/train.py -ngpus 1 -bsize 48 -fw transformer -out bertal_wkori_gdirect -layer_drop 0.0 -op adagrad -lr 0.01 --mode transbert_ori -nh 8 -nhl 6 -nel 6 -ndl 6 -lc True -eval_freq 0 --fetch_mode tf_example_dataset --subword_vocab_size 0 --dmode wk --tie_embedding all --bert_mode bert_token:bertbase:init --environment aws -warm /home/zhaos5/projs/ts/ckpt/model.ckpt-4535134 --memory direct --direct_mode gate > ori_train_gdirect.log &
CUDA_VISIBLE_DEVICES=7 nohup python ../../../model/eval.py  -ngpus 1 -bsize 8 -fw transformer -out bertal_wkori_gdirect -layer_drop 0.0 -op adagrad -lr 0.01 --mode transbert_ori -nh 8 -nhl 6 -nel 6 -ndl 6 -lc True -eval_freq 0 --subword_vocab_size 0 --dmode wk --tie_embedding all --bert_mode bert_token:bertbase:init --environment aws > ori_eval_gdirect.log &


#ori direct static_seq with npad static_seq 90149
CUDA_VISIBLE_DEVICES=3 nohup python ../../../model/train.py -ngpus 1 -bsize 12 -fw transformer -out bertal_wkori_direct_staicseq_npad -layer_drop 0.0 -op adagrad -lr 0.01 --mode transbert_ori -nh 8 -nhl 6 -nel 6 -ndl 6 -lc True -eval_freq 0 --fetch_mode tf_example_dataset --subword_vocab_size 0 --dmode wk --tie_embedding all --bert_mode bert_token:bertbase --environment aws -warm /home/zhaos5/projs/ts/perf/bertal_wkori_direct/ckpt/model.ckpt-155470 --memory direct --train_mode static_seq --npad_mode static_seq > ori_train_static_seq_npad.log &
CUDA_VISIBLE_DEVICES=4 nohup python ../../../model/eval.py -ngpus 1 -bsize 12 -fw transformer -out bertal_wkori_direct_staicseq_npad -layer_drop 0.0 -op adagrad -lr 0.01 --mode transbert_ori -nh 8 -nhl 6 -nel 6 -ndl 6 -lc True -eval_freq 0 --fetch_mode tf_example_dataset --subword_vocab_size 0 --dmode wk --tie_embedding all --bert_mode bert_token:bertbase --environment aws --memory direct --train_mode static_seq --npad_mode static_seq > ori_eval_static_seq_npad.log &


