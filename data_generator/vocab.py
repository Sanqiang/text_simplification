from data_generator.vocab_config import DefaultConfig

class Vocab:
    def __init__(self, voc_config):
        self.voc_config = (DefaultConfig()
                           if voc_config is None else voc_config)

    @staticmethod
    def ProcessWord(word, voc_config=None):
        voc_config = (DefaultConfig()
                      if voc_config is None else voc_config)

        if word:
            # All numeric will map to #
            if word[0].isnumeric() or word[0] == '+' or word[0] == '-':
                return '#'
            # Keep mark
            elif len(word) == 1 and not word[0].isalpha():
                return word
            # Actual word
            else:
                if voc_config.lower_case:
                    word = word.lower()
                return word