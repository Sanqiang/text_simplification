import tensorflow as tf

batch_size = 3
vocab_size = 5

ref = tf.get_variable('aa', [batch_size, vocab_size], initializer=tf.constant_initializer(1))
indices = tf.constant([1,])
vals = tf.zeros([batch_size, 1])

col_indices_nd = tf.stack(tf.meshgrid(tf.range(tf.shape(ref)[0]), indices, indexing='ij'), axis=-1)
var_update_cols = tf.scatter_nd_update(ref, col_indices_nd, vals)


sess = tf.Session()
sess.run(tf.initialize_all_variables())
print(sess.run(ref))
print(sess.run(var_update_cols))


# import tensorflow as tf
#
# var = tf.get_variable('var', [4, 3], tf.float32, initializer=tf.zeros_initializer())
# updates = tf.placeholder(tf.float32, [None, None])
# indices = tf.placeholder(tf.int32, [None])
# # Update rows
# # var_update_rows = tf.scatter_update(var, indices, updates)
# # Update columns
# col_indices_nd = tf.stack(tf.meshgrid(tf.range(tf.shape(var)[0]), indices, indexing='ij'), axis=-1)
# var_update_cols = tf.scatter_nd_update(var, col_indices_nd, updates)
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     # print('Rows updated:')
#     # print(sess.run(var_update_rows, feed_dict={updates: [[1, 2, 3], [4, 5, 6]], indices: [3, 1]}))
#     print('Columns updated:')
#     print(sess.run(var_update_cols, feed_dict={updates: [[1, 5], [2, 6], [3, 7], [4, 8]], indices: [0, 2]}))