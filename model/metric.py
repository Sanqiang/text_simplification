import numpy as np
from util import constant
from util.sari import SARIsent
from util.fkgl import get_fkgl
from util.decode import truncate_sent
from model.lm import GoogleLM
from model.ppdb import PPDB

from nltk.translate.bleu_score import sentence_bleu


class Metric:
    def __init__(self, model_config, data):
        self.model_config = model_config
        self.data = data

