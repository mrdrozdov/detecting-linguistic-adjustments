import os
import json
import random

from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

# from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

from embeddings import context_insensitive_character_embeddings

from read_data import *


class BatchManager(object):
    def __init__(self, pad_token=None, emb_layer=None, cuda=False):
        super(BatchManager, self).__init__()
        self.pad_token = pad_token
        self.emb_layer = emb_layer
        self.cuda = cuda

    @property
    def device(self):
        return torch.cuda.current_device() if self.cuda else None

    def __tokenize_batch(self, sentences):
        return [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' '.join(x)))
                for x in sentences]

    def __pad_and_make_tensor(self, tokenized):
        lengths = torch.tensor([len(x) for x in tokenized], dtype=torch.long, device=self.device)
        tensors = [self.emb_layer(torch.tensor(x, dtype=torch.long, device=self.device)) for x in tokenized]
        padded = torch.nn.utils.rnn.pad_sequence(tensors,
            batch_first=True, padding_value=self.pad_token)
        return torch.nn.utils.rnn.pack_padded_sequence(padded,
            batch_first=True, lengths=lengths, enforce_sorted=False)

    def prepare_tokens(self, batch_map):
        tensor = self.__pad_and_make_tensor(batch_map['sentences'])
        return tensor

    def prepare_labels(self, batch_map):
        return torch.tensor(batch_map['labels'], dtype=torch.long, device=self.device)


class Classifier(nn.Module):
    def __init__(self, embedding_size=1024, size=768, n_classes=4):
        super(Classifier, self).__init__()
        self.rnn = nn.GRU(input_size=embedding_size, hidden_size=size, batch_first=True)
        self.classify = nn.Sequential(
                nn.Linear(size, size),
                nn.ReLU(),
                nn.Linear(size, n_classes))

    def forward(self, tokens_tensor):
        _, h_n = self.rnn(tokens_tensor)
        hidden_state = h_n.squeeze(0)
        return self.classify(hidden_state)


class Summary(object):
    def __init__(self, trail=100):
        self.trail = trail
        self.history = {}

    def update_kv(self, k, v):
        self.history.setdefault(k, []).append(v)
        if self.trail is not None:
            self.history[k] = self.history[k][-self.trail:]

    def mean(self, k):
        return np.mean(self.history[k])

    def sum(self, k):
        return np.sum(self.history[k])


def print_cm(S, idx2label):
    print('--- confusion-matrix (ground_truth, predicted) ---')
    for gt in idx2label.values():
        correct = 0
        total = 0
        for pred in idx2label.values():
            k = (gt, pred)
            x = S.sum(k)
            total += x
            if gt == pred:
                correct += x
            print(k, x)
        print('{} {:.3f} ({}/{})'.format(
            gt, correct/total, correct, total))
    print('--- confusion-matrix ---')


def run_eval(options, model, batch_manager, eval_dataset):
    print('### eval ###')
    eval_iterator = BatchIterator(eval_dataset['sentences'], eval_dataset['extra'])
    idx2label = {v: k for k, v in eval_dataset['metadata']['label2idx'].items()}

    S = Summary(trail=None)

    for batch_map in eval_iterator.get_iterator(batch_size=options.batch_size, include_partial=True):
        packed_sequence = batch_manager.prepare_tokens(batch_map)
        labels_tensor = batch_manager.prepare_labels(batch_map)
        logits = model(packed_sequence)

        predictions = logits.argmax(dim=1)

        # Advanced Logging.
        batch_size = labels_tensor.shape[0]
        toupdate = {}
        for gt in idx2label.values():
            for pred in idx2label.values():
                toupdate[(gt, pred)] = 0
        for i in range(batch_size):
            pred = idx2label[predictions[i].item()]
            gt = idx2label[labels_tensor[i].item()]
            toupdate[(gt, pred)] += 1
        for gt in idx2label.values():
            for pred in idx2label.values():
                k = (gt, pred)
                v = toupdate[k]
                S.update_kv(k, v)

    print_cm(S, idx2label)


def run(options):
    num_epochs = 10

    datasets = OrderedDict()
    datasets['tr'] = AdjustmentDatasetBaseline().read(options.tr_file)
    datasets['va'] = AdjustmentDatasetBaseline().read(options.va_file)

    datasets = ConsolidateDatasets().run(datasets)

    train_dataset = datasets['tr']
    train_iterator = BatchIterator(train_dataset['sentences'], train_dataset['extra'])
    train_word2idx = train_dataset['metadata']['word2idx']
    idx2label = {v: k for k, v in train_dataset['metadata']['label2idx'].items()}
    embeddings = context_insensitive_character_embeddings(
        os.path.expanduser('~/tmp/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'),
        os.path.expanduser('~/tmp/elmo_2x4096_512_2048cnn_2xhighway_options.json'),
        word2idx=train_word2idx,
        cuda=False,
        cache_dir='./elmo_cache'
        )

    emb_layer = nn.Embedding.from_pretrained(torch.from_numpy(embeddings), freeze=True)
    batch_manager = BatchManager(
        pad_token=train_word2idx[AdjustmentDatasetBaseline.PADDING_TOKEN],
        emb_layer=emb_layer,
        cuda=options.cuda)
    batch_manager.emb_layer = emb_layer.to(batch_manager.device)

    model = Classifier(embedding_size=embeddings.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=2e-3, betas=(0.9, 0.999), eps=1e-8)
    S = Summary(trail=100)
    log_every = 100
    summary_every = 100
    eval_every = 500

    if options.cuda:
        model.cuda()

    step = 0

    for epoch in range(num_epochs):
        for batch_map in train_iterator.get_iterator(batch_size=options.batch_size):

            packed_sequence = batch_manager.prepare_tokens(batch_map)
            labels_tensor = batch_manager.prepare_labels(batch_map)
            logits = model(packed_sequence)
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

            # Standard Logging.
            batch_output = {}
            batch_output['loss'] = loss.item()
            batch_output['accuracy'] = accuracy
            for k in ['loss', 'accuracy']:
                S.update_kv(k, batch_output[k])

            if step % log_every == 0:
                print('step = {:08}, loss = {:.3f}, accuracy-mean = {:.3f}'.format(
                    step, S.mean('loss'), S.mean('accuracy')))

            # Advanced Logging.
            batch_size = labels_tensor.shape[0]
            toupdate = {}
            for gt in idx2label.values():
                for pred in idx2label.values():
                    toupdate[(gt, pred)] = 0
            for i in range(batch_size):
                pred = idx2label[predictions[i].item()]
                gt = idx2label[labels_tensor[i].item()]
                toupdate[(gt, pred)] += 1
            for gt in idx2label.values():
                for pred in idx2label.values():
                    k = (gt, pred)
                    v = toupdate[k]
                    S.update_kv(k, v)

            if step % summary_every == 0:
                print_cm(S, idx2label)

            if step % eval_every == 0:
                run_eval(options, model, batch_manager, datasets['va'])

            step += 1


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--tr_file', default='./adjustment-tr.jsonl', type=str)
    parser.add_argument('--va_file', default='./adjustment-va.jsonl', type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    options = parser.parse_args()

    if options.seed is None:
        options.seed = random.randint(0, 1e7)

    run(options)
