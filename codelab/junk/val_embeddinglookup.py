import tensorflow as tf
res = tf.nn.embedding_lookup(
    tf.zeros([3, 5, 8]),
    tf.zeros([3, 2], dtype=tf.int32))

print(res)