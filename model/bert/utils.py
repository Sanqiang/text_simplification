import tensorflow as tf
import tensorflow.contrib.slim as slim


DEFAULT_BERT_MODEL = '/zfs1/hdaqing/saz31/dataset/vocab/bert/uncased_L-24_H-1024_A-16/bert_model.ckpt'


def restore_bert(prefix='model/', ckpt='', model='bert/'):
    ckpt_path = ckpt or DEFAULT_BERT_MODEL
    var_list = slim.get_variables_to_restore()
    reader = tf.train.NewCheckpointReader(ckpt_path)
    var_dict = {var.op.name: var for var in var_list}
    available_vars = {}
    for var in var_dict:
        if not var.startswith(prefix + model):
            continue
        if reader.has_tensor('bert/' + var[len(prefix+model):]):
            available_vars['bert/' + var[len(prefix+model):]] = var_dict[var]
            print('bert/' + var[len(prefix+model):] + '=>' + var)
        else:
            if var == 'variables/embedding_complex':
                available_vars['bert/' + var[len(prefix+model):]] = var_dict[
                    prefix + model + 'embeddings/word_embeddings']
            elif var == 'variables/embedding_simple':
                available_vars['bert/' + var[len(prefix+model):]] = var_dict[
                    prefix + model + 'embeddings/word_embeddings']
            else:
                print('mismatch tensor for bert %s' % var)
                raise ValueError('mismatch tensor for bert.')

    bert_restore_ckpt = slim.assign_from_checkpoint_fn(
        ckpt_path, available_vars,
        ignore_missing_vars=False, reshape_variables=False)
    return bert_restore_ckpt


def merge_tokens(tokens):
    sent = []
    for tid, token in enumerate(tokens):
        if token.startswith('##') and tid > 0:
            sent[len(sent)-1] += token[2:]
        else:
            sent.append(token)
    return ' '.join(sent)


# if __name__ == '__main__':
#     print(merge_tokens(['###', '###']))
#     print(merge_tokens(['?', '##a', '###', '###', '###']))
#     print(merge_tokens(['i', 'am', 'pig', '##.']))
#     print(merge_tokens(['?', '##?', '##?']))
