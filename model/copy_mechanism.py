import tensorflow as tf
from util import constant


def word_distribution(decoder_logit_list, decoder_output_list, encoder_outputs, encoder_embedding,
                      sentence_complex_input_placeholder, obj_tensors, model_config, data, segment_mask=None, is_test=False):
    if model_config.architecture == 'ut2t':
        # attn_dists = obj_tensors[
        #     'model/transformer_decoder/decoder/universal_transformer_act/encdec_attention/multihead_attention/dot_product_attention']
        # attn_dists = attn_dists[:, 0, :, :]
        raise ValueError('Cannot use copy in u2t2')
    else:
        attn_dists = obj_tensors[
            'model/transformer_decoder/decoder/layer_%s/encdec_attention/multihead_attention/dot_product_attention'
            % (model_config.num_decoder_layers-1)]
        attn_dists_stack = attn_dists[:, 0, :, :]
        if is_test:
            attn_dists = [attn_dists_stack[:, -1, :]]
            attn_dists_stack = tf.expand_dims(attn_dists[0], axis=1)
        else:
            attn_dists = tf.unstack(attn_dists_stack, axis=1)

    sentence_complex_input = tf.stack(sentence_complex_input_placeholder, axis=1)
    ignore_token_idx = data.vocab_simple.encode(constant.SYMBOL_UNK)
    if type(ignore_token_idx) == list:
        assert len(ignore_token_idx) == 1
        ignore_token_idx = ignore_token_idx[0]
    if segment_mask is not None:
        sentence_complex_input *= segment_mask
        sentence_complex_input += tf.to_int32(tf.to_float(
            tf.equal(sentence_complex_input, 0)) * ignore_token_idx)

    batch_nums = tf.range(0, limit=model_config.batch_size)
    batch_nums = tf.expand_dims(batch_nums, 1)
    batch_nums = tf.tile(batch_nums, [1, model_config.max_complex_sentence])
    indices = tf.stack((batch_nums, sentence_complex_input), axis=2)
    attn_dists_projected = [tf.scatter_nd(
        indices, copy_dist, [model_config.batch_size, data.vocab_simple.vocab_size()])
        for copy_dist in attn_dists]
    for attn_id, attn_dist in enumerate(attn_dists_projected):
        mask = tf.concat([tf.ones([model_config.batch_size, ignore_token_idx]),
                tf.zeros([model_config.batch_size, 1]),
                tf.ones([model_config.batch_size, data.vocab_simple.vocab_size()-ignore_token_idx-1])], axis=1)
        attn_dists_projected[attn_id] *= mask

    attn_dists_projected = tf.stack(attn_dists_projected, axis=1)
    attn_dists_projected = tf.stop_gradient(attn_dists_projected)

    decoder_logit = tf.stack(decoder_logit_list, axis=1)
    decoder_output = tf.stack(decoder_output_list, axis=1)

    context_vectors = tf.matmul(attn_dists_stack, encoder_outputs)
    context_emb_vectors = tf.matmul(attn_dists_stack, encoder_embedding)
    context_vectors = tf.stop_gradient(context_vectors)
    context_emb_vectors = tf.stop_gradient(context_emb_vectors)
    decoder_output = tf.stop_gradient(decoder_output)
    # decoder_logit = tf.stop_gradient(decoder_logit)
    evidence = tf.concat([context_vectors, context_emb_vectors, decoder_output], axis=-1)
    gate = tf.layers.dense(evidence, 1, activation=tf.nn.sigmoid)
    if 'thres' in model_config.pointer_mode:
        output_logit = tf.cond(
            tf.greater_equal(gate, 0.5),
            lambda : attn_dists_projected,
            lambda : decoder_logit
        )
    elif 'fuse' in model_config.pointer_mode:
        output_logit = gate * attn_dists_projected + (1-gate) * decoder_logit
    else:
        raise NotImplementedError('unknown output pointer')

    return tf.unstack(output_logit, axis=1)