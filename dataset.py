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


class metaDataset(Dataset):
    def __init__(self, filename, episodes, support, shot, way, mode='train'):
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

        self.episodes = episodes
        self.need_to_generate = episodes
        self.support_per_task = support
        self.query_per_task = shot
        self.class_per_task = way
        self.mode = mode
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
        task = {
            'support_sents': support_sents,
            'support_labels': support_labels,
            'query_sents': query_sents,
            'query_labels': query_labels
        }
        return self.shuffle(task)

    def shuffle(self, task):
        support_sents = task['support_sents']
        support_labels = task['support_labels']
        support_idx = list(range(len(support_sents)))
        random.shuffle(support_idx)
        support_sents = [support_sents[i] for i in support_idx]
        support_labels = [support_labels[i] for i in support_idx]
        query_sents = task['query_sents']
        query_labels = task['query_labels']
        query_idx = list(range(len(query_sents)))
        random.shuffle(query_idx)
        query_sents = [query_sents[i] for i in query_idx]
        query_labels = [query_labels[i] for i in query_idx]
        return {
            'support_sents': support_sents,
            'support_labels': support_labels,
            'query_sents': query_sents,
            'query_labels': query_labels
        }


def meta_collate_fn(samples):
    pdb.set_trace()
    pass