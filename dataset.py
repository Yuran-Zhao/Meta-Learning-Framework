import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from collections import defaultdict
import json
import pickle
import pdb
from utils import metaBertTokenizer


class metaDataset(Dataset):
    def __init__(self, filename, tokenizer, args):
        self.tokenizer = tokenizer
        with open(filename, 'r', encoding='utf8') as fin:
            data = fin.read().split('\n')
        self.sents = []
        self.labels = []
        for line in tqdm(data):
            if len(line) == 0:
                continue
            sent, label = line.split('\t')
            self.sents.append(sent)
            self.labels.append(int(label))

        self.episodes = args.train_episodes
        self.need_to_generate = args.train_episodes
        self.support_per_task = args.support
        self.query_per_task = args.shot
        self.class_per_task = args.N_way
        self.mode = args.mode
        self.classes = list(set(self.labels))
        self.num_classes = len(self.classes)
        self.class2sents = defaultdict(list)
        self.init_class2sents()

    def init_class2sents(self):
        for c in self.classes:
            index = [idx for idx, l in enumerate(self.labels) if l == c]
            self.class2sents[c] = [self.sents[i] for i in index]

    def __len__(self):
        return self.episodes

    def __getitem__(self, index):
        task = self.generate_task()
        self.need_to_generate -= 1
        return task

    def generate_task(self):
        random.shuffle(self.classes)
        chosen_class = self.classes[:self.class_per_task]
        support_sents = []
        support_labels = []
        query_sents = []
        query_labels = []
        for c in chosen_class:
            idx = list(range(len(self.class2sents[c])))
            random.shuffle(idx)
            support_idx = idx[:self.support_per_task]
            query_idx = idx[self.support_per_task:self.support_per_task +
                            self.query_per_task]
            support_sents += [self.class2sents[c][i] for i in support_idx]
            query_sents += [self.class2sents[c][i] for i in query_idx]
            support_labels += [c] * self.support_per_task
            query_labels += [c] * self.query_per_task
        # these two inputs are dicts, each value is data of task
        support_inputs = self.tokenizer.tokenize(support_sents)
        qeury_inputs = self.tokenizer.tokenize(query_sents)
        task = {
            'support_inputs': support_inputs,
            'support_labels': support_labels,
            'query_inputs': qeury_inputs,
            'query_labels': query_labels
        }

        return self.shuffle(task)

    def shuffle(self, task):
        support_input_ids = task['support_inputs']['input_ids'].tolist()
        support_attention_mask = task['support_inputs'][
            'attention_mask'].tolist()
        if isinstance(self.tokenizer, metaBertTokenizer):
            support_token_type_ids = task['support_inputs'][
                'token_type_ids'].tolist()
        support_labels = task['support_labels']
        support_idx = list(range(len(support_labels)))
        random.shuffle(support_idx)
        support_input_ids = [support_input_ids[i] for i in support_idx]
        support_attention_mask = [
            support_attention_mask[i] for i in support_idx
        ]
        if isinstance(self.tokenizer, metaBertTokenizer):
            support_token_type_ids = [
                support_token_type_ids[i] for i in support_idx
            ]
        support_labels = [support_labels[i] for i in support_idx]
        support_inputs = {
            'input_ids': torch.tensor(support_input_ids),
            'attention_mask': torch.tensor(support_attention_mask),
        }
        if isinstance(self.tokenizer, metaBertTokenizer):
            support_inputs['token_type_ids'] = torch.tensor(
                support_token_type_ids)
        support_labels = torch.tensor(support_labels)

        query_input_ids = task['query_inputs']['input_ids'].tolist()
        query_attention_mask = task['query_inputs']['attention_mask'].tolist()
        if isinstance(self.tokenizer, metaBertTokenizer):
            query_token_type_ids = task['query_inputs'][
                'token_type_ids'].tolist()
        query_labels = task['query_labels']
        query_idx = list(range(len(query_labels)))
        random.shuffle(query_idx)
        query_input_ids = [query_input_ids[i] for i in query_idx]
        query_attention_mask = [query_attention_mask[i] for i in query_idx]
        if isinstance(self.tokenizer, metaBertTokenizer):
            query_token_type_ids = [query_token_type_ids[i] for i in query_idx]
        query_labels = [query_labels[i] for i in query_idx]
        query_inputs = {
            'input_ids': torch.tensor(query_input_ids),
            'attention_mask': torch.tensor(query_attention_mask),
        }
        if isinstance(self.tokenizer, metaBertTokenizer):
            query_inputs['token_type_ids'] = torch.tensor(query_token_type_ids)
        query_labels = torch.tensor(query_labels)

        rets = {
            'support_inputs': support_inputs,
            'support_labels': support_labels,
            'query_inputs': query_inputs,
            'query_labels': query_labels
        }
        return rets


def meta_collate_fn(samples):
    pass