class DefaultConfig():
    batch_size = 25
    dimension = 512
    max_complex_sentence = 30
    max_simple_sentence = 20

    optimizer = 'adagrad'
    learning_rate = 0.001
    max_grad_staleness = 0.0
    max_grad_norm = 4.0