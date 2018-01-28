import os
import psutil
import tensorflow as tf

x = tf.constant(1, shape=(1000,1000,1000))
sess = tf.Session()
print(sess.run(x))
pid = os.getpid()
py = psutil.Process(pid)
memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
print('memory use:', memoryUse)
