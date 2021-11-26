import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from model import MLPClassifier


class metaBert(nn.Module):
    def __init__(self, args):
        config = BertConfig.from_pretrained(args.bert_path)
        self.bert = BertModel.from_pretrained(args.bert_path)
        self.classifier = MLPClassifier(config.hidden_size,
                                        args.total_class_num)

    def forward(self, inputs):
        bert_output = self.bert(**inputs)
        logits = self.classifier(bert_output)
        return logits
