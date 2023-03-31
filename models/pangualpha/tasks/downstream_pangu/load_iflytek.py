
import os

import itertools
import numpy as np
import json
from megatron import get_tokenizer
from megatron.tokenizer import JIEBATokenizer

from megatron.utils import get_ltor_masks_and_position_ids


class iflytek_dataset(object):
    def __init__(self):
        self.zc_cache = []
        self.data_path = '/userhome/dataset/cluedatasets/iflytek_public/'
        vocab_file = '/userhome/pclproject/gpt/Megatron-LM-1.1-Pangu/megatron/tokenizer/bpe_4w_pcl/vocab'

        self.tokenizer = get_tokenizer()
        # self.tokenizer = JIEBATokenizer(vocab_file)
        self.seq_length = 1024
        self.num_samples = 0
        print()

    def load_iflytek_train_example_for_shot(self, data_path, num_sample=2, np_rng=None, max_len=None, input_str_format=None):
        if input_str_format is None:
            input_str_format = "这是关于{label}的应用程序：{sentence}"
        # input_str_format = "{s}：{label}"
        if np_rng is None:
            np_rng = np.random.default_rng()
        if len(self.zc_cache)>0:
            z0 = self.zc_cache[0]
        else:
            tmp0 = [os.path.join(data_path,x) for x in os.listdir(data_path) if x.startswith('train')]
            assert len(tmp0)==1
            train_file = tmp0[0]
            with open(train_file, 'r', encoding='utf-8') as fid:
                z0 = [json.loads(x) for x in fid.readlines()]
            self.zc_cache.append(z0)

        # select sample with balanced labels
        hf0 = lambda x: x[1]
        tmp0 = itertools.groupby(sorted([(x,y['label_des']) for x,y in enumerate(z0)],key=hf0), key=hf0)
        group_index = [np.array([z[0] for z in y]) for x,y in tmp0]
        for x in group_index:
            np_rng.shuffle(x) #in-place
        tmp0 = (num_sample-1)//len(group_index) + 1
        tmp1 = np.concatenate([x[:tmp0] for x in group_index])
        np_rng.shuffle(tmp1)
        selected_index = tmp1[:num_sample]
        # selected_index = np_rng.permutation(len(z0))[:num_sample]

        examples = []
        for x in selected_index:
            sentence = z0[x]['sentence'] if max_len is None else z0[x]['sentence'][:max_len]
            tmp0 = input_str_format.format(label=z0[x]['label_des'], sentence=sentence)
            examples.append(tmp0)
        ret = {
            'zero_shot': '',
            'one_shot': examples[0]+'\n',
            'few_shot':('\n'.join(examples)) + '\n',
        }
        return ret

    def getSamples(self, para_config):
        with open(os.path.join(self.data_path, 'train.json'), "r", encoding="utf-8") as fid:
            ground_truth = [json.loads(x) for x in fid]
        id_to_label = {int(x['label']):x['label_des'] for x in ground_truth}

        task = para_config['task']
        max_len = para_config['max_len']
        tag_new_example = para_config['tag_new_example']
        few_shot_num_sample = para_config['few_shot_num_sample']
        np_seed = para_config['np_seed']
        new_mask = para_config['new_mask']
        input_str_format = para_config['input_str_format']
        input_str_format_mask = input_str_format.rsplit('{',1)[0]
        input_str_format_mask_tag_label = '{label}' in input_str_format_mask
        np_seed = para_config['np_seed']
        np_rng = np.random.default_rng(seed=np_seed) #must be same across model-parallel
        with open(os.path.join(self.data_path, 'dev.json'), "r", encoding="utf-8") as fid:
            tmp0 = [json.loads(x) for x in fid] #[:200]
            ground_truth = [tmp0[x] for x in np_rng.permutation(len(tmp0))]

        print('total num samples of iflytek dev dataset is : ',len(ground_truth))
        z0 = []
        zc_print_ind = 0
        if not tag_new_example:
            shot_to_example = self.load_iflytek_train_example_for_shot(self.data_path, num_sample=few_shot_num_sample,
                                                                  np_rng=np_rng, max_len=max_len, input_str_format=input_str_format)
            example = shot_to_example[task]
        for instance in ground_truth:
            zc_print_ind += 1
            if tag_new_example:
                shot_to_example = self.load_iflytek_train_example_for_shot(self.data_path, num_sample=few_shot_num_sample,
                                                                      np_rng=np_rng, max_len=max_len, input_str_format=input_str_format)
                example = shot_to_example[task]

            true_label = instance['label_des']
            tmp0 = sorted(list(set(id_to_label.values()) - {true_label}))
            fake_label = [tmp0[x] for x in np_rng.permutation(len(tmp0))[:3]] #[:119]
            instance_tf_label = [true_label] + fake_label
            instance_tf_label = [instance_tf_label[x] for x in np_rng.permutation(len(instance_tf_label))] #shuffle
            input_ids_list = []
            loss_mask_list = []
            label_list = []
            input_str_list = []
            for label_i in instance_tf_label:
                if input_str_format_mask_tag_label:
                    tmp0 = example + input_str_format_mask.format(label=label_i)
                else:
                    tmp0 = example + input_str_format_mask.format(sentence=instance['sentence'])
                tmp0 = self.tokenizer.tokenize(tmp0)
                tmp1 = example + input_str_format.format(label=label_i, sentence=instance['sentence'])

                input_ids = self.tokenizer.tokenize(tmp1)[:self.seq_length]
                input_str_list.append(tmp1)

                # tmp0 = tokenizer.tokenize(f"{example}{instance['sentence']}")
                # input_ids = tokenizer.tokenize(f"{example}{instance['sentence']}：{label_i}")[:config.seq_length]

                mask = np.zeros(self.seq_length)
                mask[len(tmp0):len(input_ids)] = 1
                # mask[:len(input_ids)] = 1
                input_ids = np.pad(input_ids, ((0,self.seq_length+1-len(input_ids)),), 'constant', constant_values=(0,self.tokenizer.pad_id))
                input_ids_list.append(input_ids)
                loss_mask_list.append(mask)
                label_list.append(label_i)

            tmp0 = input_ids_list
            tmp1 = loss_mask_list
            label = [x for x,y in enumerate(label_list) if y==true_label][0]
            yield {'input_ids_list':tmp0,
                   "loss_mask_list":tmp1,
                   'label':label
                   }
            # yield {'input_ids_list':tmp0,
            #        "loss_mask_list":tmp1,
            #        'label':label,
            #        'input_str_list':input_str_list
            #        }
            # z0.append((tmp0,tmp1,label,input_str_list))


if __name__ == '__main__':

    iflytek = iflytek_dataset()
    samples = iflytek.getSamples()

    for sample in samples:
        print()
    pass