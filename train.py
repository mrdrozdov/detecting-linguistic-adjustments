import os
import json
import random

import torch
import torch.nn as nn
import torch.optim as optim

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
        
    def prepare_tokens(self, batch_map):
        tokenized = self.__tokenize_batch(batch_map['sentences'])
        tensor = self.__pad_and_make_tensor(tokenized)
        return tensor

    def prepare_labels(self, batch_map):
        return torch.tensor(batch_map['labels'], dtype=torch.long)


class Classifier(nn.Module):
    def __init__(self, size=768, n_classes=4):
        super(Classifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classify = nn.Linear(size, n_classes)

    def forward(self, tokens_tensor):
        encoded_layers, _ = self.bert(tokens_tensor, None)
        hidden_state = encoded_layers[-1][:, 0]
        return self.classify(hidden_state)


def run(options):
    num_epochs = 10

    train_dataset = AdjustmentDataset().read(options.train_file)
    train_iterator = BatchIterator(train_dataset['sentences'], train_dataset['extra'])

    batch_manager = BatchManager()

    model = Classifier()
    optimizer = optim.Adam(model.parameters(), lr=2e-3, betas=(0.9, 0.999), eps=1e-8)

    for epoch in range(num_epochs):
        for batch_map in train_iterator.get_iterator():

            tokens_tensor = batch_manager.prepare_tokens(batch_map)
            labels_tensor = batch_manager.prepare_labels(batch_map)

            logits = model(tokens_tensor)

            loss = nn.CrossEntropyLoss()(logits, labels_tensor)

            # Compute gradient.
            optimizer.zero_grad()
            loss.backward()
            # Clip gradient.
            params = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            # Update weights.
            optimizer.step()

            predictions = logits.argmax(dim=1)
            accuracy = torch.sum(predictions == labels_tensor).float().item() / labels_tensor.shape[0]

            print('loss = {:.3f}, accuracy = {:.3f}'.format(loss.item(), accuracy))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--train_file', default='./adjustment-data.jsonl', type=str)
    options = parser.parse_args()

    if options.seed is None:
        options.seed = random.randint(0, 1e7)

    run(options)
