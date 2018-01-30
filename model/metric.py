import numpy as np
from util import constant
from util.sari import SARIsent, WeightedSARIsent
from util.fkgl import get_fkgl
from util.decode import truncate_sent
from model.lm import GoogleLM
from model.ppdb import PPDB

from nltk.translate.bleu_score import sentence_bleu


class Metric:
    def __init__(self, model_config, data):
        self.model_config = model_config
        self.data = data

    def self_crititcal_reward(self, sample_target_list, greed_target_list, gt_simp_list, gt_comp_list):
        rewards = []
        batch_size = np.shape(gt_simp_list)[0]
        num_steps = np.shape(gt_simp_list)[1]
        for batch_i in range(batch_size):
            if 'sari' in self.model_config.rl_configs:
                cur_sample_target_list = ' '.join([str(it) for it in sample_target_list[batch_i]])
                cur_greed_target_list = ' '.join([str(it) for it in greed_target_list[batch_i]])
                cur_gt_simp_list = ' '.join([str(it) for it in gt_simp_list[batch_i]])
                cur_gt_comp_list = ' '.join([str(it) for it in gt_comp_list[batch_i]])

                reward_sample = SARIsent(cur_gt_comp_list, cur_sample_target_list, [cur_gt_simp_list])
                reward_greed = SARIsent(cur_gt_comp_list, cur_greed_target_list, [cur_gt_simp_list])
                reward = [reward_sample-reward_greed for _ in range(num_steps)]

                # if 'sari_ppdb_simp_weight' in self.model_config.rl_configs:
                #     weight = self.model_config.rl_configs['sari_ppdb_simp_weight']
                #

                rewards.append(reward)
            else:
                raise Exception('No RL metric provided.')
        return np.array(rewards, dtype=np.float32)



