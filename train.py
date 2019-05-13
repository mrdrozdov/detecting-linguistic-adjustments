import os
import json
import random

import torch

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

from read_data import *


class BatchManager(object):
    def __init__(self):
        super(BatchManager, self).__init__()

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __tokenize_batch(self, sentences):
        return [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' '.join(x)))
                for x in sentences]

    def __pad_and_make_tensor(self, tokenized):
        max_length = max([len(x) for x in tokenized])
        xs = [x + [0] * (max_length - len(x)) for x in tokenized]
        return torch.tensor(xs, dtype=torch.long)
        
    def prepare_batch(self, batch_map):
        tokenized = self.__tokenize_batch(batch_map['sentences'])
        tensor = self.__pad_and_make_tensor(tokenized)
        return tensor


def run(options):
    num_epochs = 10

    train_dataset = AdjustmentDataset().read(options.train_file)
    train_iterator = BatchIterator(train_dataset['sentences'], train_dataset['extra'])

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    batch_manager = BatchManager()

    for epoch in range(num_epochs):
        for batch_map in train_iterator.get_iterator():

            model_input = batch_manager.prepare_batch(batch_map)

            print(model_input.shape)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--train_file', default='./adjustment-data.jsonl', type=str)
    options = parser.parse_args()

    if options.seed is None:
        options.seed = random.randint(0, 1e7)

    run(options)
