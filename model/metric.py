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

    def self_crititcal_reward(self, sample_target_list, greed_target_list, gt_simp_list, gt_comp_list,
                              rule_target_input_placeholder):
        rewards = []
        batch_size = np.shape(gt_simp_list)[0]
        num_steps = np.shape(gt_simp_list)[1]
        for batch_i in range(batch_size):
            reward = [1.0 for _ in range(num_steps)]
            cur_sample_target_list = sample_target_list[batch_i]
            cur_greed_target_list = greed_target_list[batch_i]
            cur_gt_simp_list = gt_simp_list[batch_i]
            cur_gt_comp_list = gt_comp_list[batch_i]
            if 'sari' in self.model_config.rl_configs:
                cur_sample_target_str = ' '.join([str(o) for o in cur_sample_target_list])
                cur_greed_target_str = ' '.join([str(o) for o in cur_greed_target_list])
                cur_gt_simp_str = ' '.join([str(o) for o in cur_gt_simp_list])
                cur_gt_comp_str = ' '.join([str(o) for o in cur_gt_comp_list])
                reward_sample = SARIsent(cur_gt_comp_str, cur_sample_target_str, [cur_gt_simp_str])
                reward_greed = SARIsent(cur_gt_comp_str, cur_greed_target_str, [cur_gt_simp_str])
                reward = [r * (reward_sample-reward_greed) for r in reward]
            if 'rule' in self.model_config.rl_configs:
                rule_weight = self.model_config.rl_configs['rule_weight']
                reward_sample = [1.0 for _ in range(num_steps)]
                reward_greed = [1.0 for _ in range(num_steps)]
                cur_rule_target_input_placeholder = rule_target_input_placeholder[batch_i]
                for step in range(num_steps):
                    if cur_sample_target_list[step] in cur_rule_target_input_placeholder:
                        reward_sample[step] *= rule_weight
                    if cur_greed_target_list[step] in cur_rule_target_input_placeholder:
                        reward_greed[step] *= rule_weight
                reward = [r * (reward_sample[step] - reward_greed[step]) for step, r in enumerate(reward)]

            rewards.append(reward)
        return np.array(rewards, dtype=np.float32)



