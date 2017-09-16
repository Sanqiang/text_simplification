import tensorflow as tf

def get_session_config(model_config):
    if not model_config.use_gpu:
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
    else:
        config = tf.ConfigProto()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = model_config.allow_growth
    config.gpu_options.per_process_gpu_memory_fraction = model_config.per_process_gpu_memory_fraction
    return config