import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


class metaBert(nn.Module):
    def __init__(self, args):
        config = BertConfig.from_pretrained(args.bert_path)
        self.bert = BertModel.from_pretrained(args.bert_path)