import tensorflow as tf

attn_dists_stack = tf.constant([
    [[1., 0., 0.], [0, 1., 0], [0, 0, 1.], [1., 0, 0]],
    [[0.4, 0.4, 0.2], [0.9, 0.1, 0], [0, 0, 1.], [1., 0, 0]]], tf.float32)
sentence_complex_input = tf.constant([[1,2,3], [4,5,5]])
attn_dists = tf.unstack(attn_dists_stack, axis=1)

batch_nums = tf.range(0, limit=2)
batch_nums = tf.expand_dims(batch_nums, 1)
batch_nums = tf.tile(batch_nums, [1, 3])
indices = tf.stack((batch_nums, sentence_complex_input), axis=2)
attn_dists_projected = [tf.scatter_nd(
    indices, copy_dist, [2, 8])
    for copy_dist in attn_dists]
for attn_id, attn_dist in enumerate(attn_dists_projected):
    mask = tf.concat([tf.ones([2, 1]),
                      tf.zeros([2, 1]),
                      tf.ones([2, 8 - 1 - 1])],
                     axis=1)
    attn_dists_projected[attn_id] *= mask

attn_dists_projected = tf.stack(attn_dists_projected, axis=1)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
print(sess.run(attn_dists_projected))