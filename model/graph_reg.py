from tensor2tensor.models import transformer

class Graph:
    def __init__(self, data, is_train, model_config):
        self.model_config = model_config
        self.data = data
        self.is_train = is_train
        self.hparams = transformer.transformer_base()
        self.setup_hparams()