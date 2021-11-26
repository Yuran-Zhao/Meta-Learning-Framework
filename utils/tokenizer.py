from transformers import BertTokenizer
from subword_nmt.apply_bpe import BPE
from .dictionary import Dictionary
import torch


class metaBertTokenizer(object):
    def __init__(self, cache_path):
        self.tokenizer = BertTokenizer.from_pretrained(cache_path)

    def tokenize(self, lines):
        # pad to the longest sequence in the batch
        # truncate to a maximum length specified by the model
        results = self.tokenizer(lines,
                                 padding=True,
                                 truncation=True,
                                 return_tensors='pt')
        return results


class metaBPETokenizer(object):
    def __init__(self, code_file, dict_file):
        with open(code_file, 'r') as fin:
            self.bpe = BPE(fin)
        self.dict = Dictionary.load(dict_file)

    def tokenize(self, lines):
        bpe_encoded = [self.bpe.process_line(line) for line in lines]
        max_length = max([len(line.split()) for line in bpe_encoded])
        encoded_lines = []
        for line in bpe_encoded:
            encoded = self.dict.encode_line(line)
            # self.dict will add an <eos> token
            encoded += [self.dict.pad] * (max_length + 1 - len(encoded))
            encoded_lines.append(encoded)
        encoded_lines = torch.tensor(encoded_lines)
        return {
            'input_ids': encoded_lines,
            'attention_mask': (encoded_lines != self.dict.pad).int()
        }
