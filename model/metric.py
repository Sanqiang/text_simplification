import numpy as np
from util import constant
from util.sari import SARIsent
from util.fkgl import get_fkgl
from model.lm import GoogleLM
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.translate.bleu_score import sentence_bleu

from collections import defaultdict


class Metric:
    def __init__(self, model_config, data):
        self.model_config = model_config
        self.data = data

        self.stopWords = set(stopwords.words('english'))
        if 'ext_simple' in self.model_config.rl_configs:
            self.gt_simple_list_ext = []
            for line in open(self.model_config.train_dataset_simple_ext):
                self.gt_simple_list_ext.append(line.strip().lower().split('\t'))
            print('Load ext_simple from %s' % self.model_config.train_dataset_simple_ext)

        if 'rule' in self.model_config.rl_configs:
            self.mappers_ori_lines, self.mappers_tar_lines = [],[]
            train_dataset_complex_ppdb = self.model_config.train_dataset_complex_ppdb
            for line in open(train_dataset_complex_ppdb):
                rules = line.strip().split('\t')
                mapper_ori, mapper_tar = defaultdict(dict), defaultdict(dict)
                for rule in rules:
                    items = rule.split('=>')
                    if len(items) == 4:
                        ori = items[1]
                        tar = items[2]
                        weight = float(items[3])
                        if self.data.vocab_simple.encode(
                                ori) >= constant.REVERED_VOCAB_SIZE and self.data.vocab_simple.encode(
                                tar) >= constant.REVERED_VOCAB_SIZE and weight > 0.0:
                            mapper_ori[ori][tar] = weight
                            mapper_tar[tar][ori] = weight
                self.mappers_ori_lines.append(mapper_ori)
                self.mappers_tar_lines.append(mapper_tar)
            print('Load line rules from %s.' % (train_dataset_complex_ppdb))

    def truncate_sent(self, sent, bos=3, eos=4):
        sent = list(sent)
        if eos in sent:
            eos = sent.index(eos)
            sent = sent[:eos]
        s_i = 0
        if len(sent) > 0 and sent[s_i] == bos:
            while s_i + 1 < len(sent) and sent[s_i + 1] == bos:
                s_i += 1
            sent = sent[s_i + 1:]
        return sent

    def self_crititcal_reward_unitv2(self, ids, step, greed_target,
                                   gt_simp_list, gt_comp_list, global_step):

        rewards = []
        sampled_ids = []
        batch_size = np.shape(gt_simp_list)[0]
        for batch_i in range(batch_size):
            base_id = 0
            base_reward = 0.0

            greed_target_str = self.data.vocab_simple.describe(greed_target[batch_i])
            if step == 0:
                sampled_ids.append(base_id)
                rewards.append(base_reward)
                continue

            id = ids[batch_i]
            mappers_ori_line = self.mappers_ori_lines[id]
            mappers_tar_line = self.mappers_tar_lines[id]
            if greed_target_str in mappers_ori_line:
                tars = mappers_ori_line[greed_target_str]
                tar = list(tars.keys())[0]

                weight = tars[tar]

                sampled_id = self.data.vocab_simple.encode(tar)
                sampled_ids.append(sampled_id)
                rewards.append(weight)

                # cur_gt_comp_list = gt_comp_list[batch_i]
                # cur_gt_comp_str = ' '.join([self.data.vocab_complex.describe(o) for o in cur_gt_comp_list])
                # print('1sampled_id:%s\tgreed_id%s\nreward:%s\nid:%s\ncur_gt_comp_str:%s\n' %
                #       (self.data.vocab_simple.describe(sampled_ids[-1]), greed_target_str, rewards[-1], id, cur_gt_comp_str))

                continue
            elif greed_target_str in mappers_tar_line and 'noneg' not in self.model_config.rl_configs:
                oris = mappers_tar_line[greed_target_str]
                ori = list(oris.keys())[0]

                weight = oris[ori]
                sampled_id = self.data.vocab_simple.encode(ori)
                sampled_ids.append(sampled_id)
                rewards.append(-1 * weight)

                # cur_gt_comp_list = gt_comp_list[batch_i]
                # cur_gt_comp_str = ' '.join([self.data.vocab_complex.describe(o) for o in cur_gt_comp_list])
                # print('2sampled_id:%s\tgreed_id%s\nreward:%s\nid:%s\ncur_gt_comp_str:%s\n' %
                #       (self.data.vocab_simple.describe(sampled_ids[-1]), greed_target_str, rewards[-1], id,
                #        cur_gt_comp_str))

                continue

            sampled_ids.append(base_id)
            rewards.append(base_reward)

        return np.array(sampled_ids, dtype=np.int32), np.array(rewards, dtype=np.float32)

    def self_crititcal_reward_unit(self, ids, step, sample_target, greed_target,
                                   gt_simp_list, gt_comp_list,
                                   global_step):

        sampled_ids = []
        rewards = []
        batch_size = np.shape(gt_simp_list)[0]
        for batch_i in range(batch_size):
            id = ids[batch_i]
            reward = 1.0

            if sample_target[batch_i] == greed_target[batch_i] or step == 0:
                rewards.append(0.0)
                sampled_ids.append(0)
                continue

            if 'sari' in self.model_config.rl_configs:
                cur_sample_target_list = self.truncate_sent(
                    np.append(gt_simp_list[batch_i][:step], sample_target[batch_i]))
                cur_greed_target_list = self.truncate_sent(
                    np.append(gt_simp_list[batch_i][:step], greed_target[batch_i]))

                base_str = self.truncate_sent(
                    [self.data.vocab_simple.describe(o) for o in gt_simp_list[batch_i][:step]])
                cur_gt_simp_list = self.truncate_sent(gt_simp_list[batch_i])
                cur_gt_comp_list = self.truncate_sent(gt_comp_list[batch_i])
                if 'sari_weight' in self.model_config.rl_configs:
                    sari_weight = self.model_config.rl_configs['sari_weight']
                cur_sample_target_str = ' '.join([self.data.vocab_simple.describe(o) for o in cur_sample_target_list])
                cur_greed_target_str = ' '.join([self.data.vocab_simple.describe(o) for o in cur_greed_target_list])
                cur_gt_simp_str = ' '.join([self.data.vocab_simple.describe(o) for o in cur_gt_simp_list])
                cur_gt_comp_str = ' '.join([self.data.vocab_complex.describe(o) for o in cur_gt_comp_list])

                try:
                    reward_base = sari_weight * SARIsent(
                        cur_gt_comp_str, base_str, [cur_gt_simp_str]) + \
                                  (1 - sari_weight) * SARIsent(
                        cur_gt_comp_str, cur_gt_simp_str, [base_str])
                except:
                    reward_base = 0.0
                try:
                    reward_sample = sari_weight * SARIsent(
                        cur_gt_comp_str, cur_sample_target_str,[cur_gt_simp_str]) +\
                                    (1-sari_weight) * SARIsent(
                        cur_gt_comp_str, cur_gt_simp_str, [cur_sample_target_str]) - reward_base
                except ZeroDivisionError:
                    reward_sample = 0.0
                try:
                    reward_greed = sari_weight * SARIsent(
                        cur_gt_comp_str, cur_greed_target_str, [cur_gt_simp_str]) +\
                                   (1-sari_weight) * SARIsent(
                        cur_gt_comp_str, cur_gt_simp_str, [cur_greed_target_str]) - reward_base
                except ZeroDivisionError:
                    reward_greed = 0.0

                reward = reward * max(reward_sample-reward_greed, 0)

                # if reward_greed < reward_sample:
                #     sample_sent = cur_sample_target_str
                #     greed_sent = cur_greed_target_str
                #     simp_sent = cur_gt_simp_str
                #     comp_sent = cur_gt_comp_str
                    # print('step:%s\nsample_sent:%s\ngreed_sent:%s\nsimp_sent:%s\ncomp_sent:%s\next:%s\nreward:%s\n=====\n' %
                    #       (step, sample_sent, greed_sent, simp_sent, comp_sent,
                    #        [cur_gt_simp_str] + self.gt_simple_list_ext[id], reward))

            if 'rule' in self.model_config.rl_configs:
                mappers_ori, mappers_tar = self.mappers_ori_lines[id], self.mappers_tar_lines[id]
                sample_target_str = self.data.vocab_simple.describe(sample_target[batch_i])
                greed_target_str = self.data.vocab_simple.describe(greed_target[batch_i])
                reward = 0.0
                sample_id = 0
                if sample_target_str in mappers_tar:
                    oris = mappers_tar[sample_target_str]
                    if greed_target_str in oris:
                        reward = oris[greed_target_str]
                        sample_id = sample_target[batch_i] # gt_simp_list[batch_i][step]
                        # print('+ %s\t%s\t%s\t%s' % (sample_target_str, greed_target_str, self.freq[k], reward))

                if sample_target_str in mappers_ori and 'noneg' not in self.model_config.rl_configs:
                    tars = mappers_ori[sample_target_str]
                    if greed_target_str in tars:
                        reward = -tars[greed_target_str]
                        sample_id = sample_target[batch_i] #gt_simp_list[batch_i][step]
                        # print('- %s\t%s\t%s\t%s' % (sample_target_str, greed_target_str, self.freq[k], reward))

                # if reward != 0:
                #     str = 'sample_target_str:%s, greed_target_str:%s\n' % (sample_target_str, greed_target_str)
                #     print(str)
            sampled_ids.append(sample_id)
            rewards.append(reward)
        return np.array(sampled_ids, dtype=np.int32), np.array(rewards, dtype=np.float32)

    def self_crititcal_reward(self, ids, sample_target_list, greed_target_list, gt_simp_list, gt_comp_list,
                              rule_target_input_placeholder):
        rewards = []
        batch_size = np.shape(gt_simp_list)[0]
        num_steps = np.shape(gt_simp_list)[1]
        for batch_i in range(batch_size):
            id = ids[batch_i]
            reward = [1.0 for _ in range(num_steps)]
            cur_sample_target_list = self.truncate_sent(sample_target_list[batch_i])
            cur_greed_target_list = self.truncate_sent(greed_target_list[batch_i])
            cur_gt_simp_list = self.truncate_sent(gt_simp_list[batch_i])
            cur_gt_comp_list = self.truncate_sent(gt_comp_list[batch_i])

            if 'sari' in self.model_config.rl_configs:
                if 'sari_weight' in self.model_config.rl_configs:
                    sari_weight = self.model_config.rl_configs['sari_weight']
                cur_sample_target_str = ' '.join([self.data.vocab_simple.describe(o) for o in cur_sample_target_list])
                cur_greed_target_str = ' '.join([self.data.vocab_simple.describe(o) for o in cur_greed_target_list])
                cur_gt_simp_str = ' '.join([self.data.vocab_simple.describe(o) for o in cur_gt_simp_list])
                cur_gt_comp_str = ' '.join([self.data.vocab_complex.describe(o) for o in cur_gt_comp_list])

                try:
                    reward_sample = sari_weight * SARIsent(cur_gt_comp_str, cur_sample_target_str, [cur_gt_simp_str]) + (1-sari_weight) * SARIsent(cur_gt_comp_str, cur_gt_simp_str, [cur_sample_target_str])
                except ZeroDivisionError:
                    reward_sample = 0.0
                try:
                    reward_greed = sari_weight * SARIsent(cur_gt_comp_str, cur_greed_target_str, [cur_gt_simp_str]) + (1-sari_weight) * SARIsent(cur_gt_comp_str, cur_gt_simp_str, [cur_greed_target_str])
                except ZeroDivisionError:
                    reward_greed = 0.0

                reward = [r * max(reward_sample-reward_greed,0) for r in reward]

                # if reward_greed < reward_sample:
                #     sample_sent = cur_sample_target_str
                #     greed_sent = cur_greed_target_str
                #     simp_sent = cur_gt_simp_str
                #     comp_sent = cur_gt_comp_str
                    # print('sample_sent:%s\ngreed_sent:%s\nsimp_sent:%s\ncomp_sent:%s\next:%s\nreward:%s\n=====\n' %
                          # (sample_sent, greed_sent, simp_sent, comp_sent, [cur_gt_simp_str] + self.gt_simple_list_ext[id],  reward))

            rewards.append(reward)
        return np.array(rewards, dtype=np.float32)

# if __name__ == '__main__':
#     ssent = "About 95 species are currently accepted .".split()
#     csent1 = "About 95 you now get in .".split()
#     csent2 = "About 95 species are now agreed .".split()
#     csent3 = "About 95 species are currently agreed .".split()
#     rsents = ["About 95 species are now known .".split()]
#
#     def reward_n(ssent, csent, rsent, n):
#         base = SARIsent()
#
#     print(SARIsent(ssent, csent1, rsents))



