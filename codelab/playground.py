import tensorflow as tf

x = tf.constant([[1,2,3],[4,5,6]])
y = tf.constant([[2],[4]])
z = x / y
with tf.Session() as sess:
    print(sess.run(z))