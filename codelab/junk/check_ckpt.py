import tensorflow as tf

PATH = '/Users/sanqiang/git/ts/tmp/model.ckpt-7882'
reader = tf.train.NewCheckpointReader(PATH)

var2shape = reader.get_variable_to_shape_map()
print(var2shape)

mem_counter = reader.get_tensor('variables/mem_counter')
print(mem_counter)

