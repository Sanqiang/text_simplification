from model.model_config import DefaultConfig
from data_generator.data import Data

def get_graph_data(data,
                   sentence_simple_input,
                   sentence_complex_input,
                   model_config=None):
    model_config = (DefaultConfig()
                    if model_config is None else model_config)

    tmp_sentence_simple, tmp_sentence_complex = [],[]
    for i in range(model_config.batch_size):
        sentence_simple, sentence_complex = data.get_data_sample()
        tmp_sentence_simple.append(sentence_simple)
        tmp_sentence_complex.append(sentence_complex)

def create_model():
    return None


if __name__ == '__main__':
    from data_generator.data import Data
    data = Data('../data/dummy_simple_dataset', '../data/dummy_complex_dataset',
                '../data/dummy_simple_vocab', '../data/dummy_complex_vocab')
    get_graph_data(data, None, None)