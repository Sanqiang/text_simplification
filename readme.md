# Integrating Transformer and Paraphrase Rules for Sentence Simplification
Paper Link: http://www.aclweb.org/anthology/D18-1355


## Note that some improvement from original EMNLP paper: 
- we modified the code to allow supporting subword and the model performs well.
- we found replacing name entities might not be a good idea (i.e. replace John to person0) since it lose some information. Instead, subword is helpful for reducing the huge vocabulary coming from name entities.
- we found the context(memory) addressing is probably redundant. Without it, the model can achieve same(even better) performance.

## Data Download:
https://drive.google.com/open?id=132Jlza-16Ws1DJ7h4O89TyxJiFSFAPw7

## Pretrained Model Download:
https://drive.google.com/open?id=16gO8cLXttGR64_xvLHgMwgJeB1DzT93N

## Command to run the model:
python model/train.py -ngpus 1 -bsize 64 -fw transformer -out bertal_wkori_direct -op adagrad -lr 0.01 --mode transbert_ori -nh 8 -nhl 6 -nel 6 -ndl 6 -lc True -eval_freq 0 --fetch_mode tf_example_dataset --subword_vocab_size 0 --dmode wk --tie_embedding all --bert_mode bert_token:bertbase:init --environment aws --memory direct
python model/eval.py  -ngpus 1 -bsize 256 -fw transformer -out bertal_wkori_direct -op adagrad -lr 0.01 --mode transbert_ori -nh 8 -nhl 6 -nel 6 -ndl 6 -lc True -eval_freq 0 --subword_vocab_size 0 --dmode wk --tie_embedding all --bert_mode bert_token:bertbase:init --environment aws

### Arugument instruction
- bsize: batch size
- out: the output folder will contains log, best model and result report
- tie_embedding: all means tie the encoder/decoder/projection w embedding, we found it can speed up the training
- bert_mode: the mode of using BERT bert_token indicates we use the subtoken vocabulary from BERT; bertbase indicates we use BERT base version (due to the memory issue, we did not try BERT large version yet)
- environment: the path config of the experiment. Please change it in model/model_config.py to fit to your system


More config you can check them in util/arguments.py


## Citation
Zhao, Sanqiang, et al. "Integrating Transformer and Paraphrase Rules for Sentence Simplification." Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. 2018.

```
@article{zhao2018integrating,
  title={Integrating Transformer and Paraphrase Rules for Sentence Simplification},
  author={Zhao, Sanqiang and Meng, Rui and He, Daqing and Andi, Saptono and Bambang, Parmanto},
  journal={arXiv preprint arXiv:1810.11193},
  year={2018}
}
```
