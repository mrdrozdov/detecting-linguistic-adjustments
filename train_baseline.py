import os
import json
import random

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
        self.classify = nn.Linear(size, n_classes)

    def forward(self, tokens_tensor):
        _, h_n = self.rnn(tokens_tensor)
        hidden_state = h_n.squeeze(0)
        return self.classify(hidden_state)


def run(options):
    num_epochs = 10

    train_dataset = AdjustmentDatasetBaseline().read(options.train_file)
    train_iterator = BatchIterator(train_dataset['sentences'], train_dataset['extra'])
    train_word2idx = train_dataset['metadata']['word2idx']
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
    history = {}
    history['accuracy'] = []
    history['loss'] = []
    trail = 100

    summary_every = 100
    summary = []

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

            batch_output = {}
            batch_output['loss'] = loss.item()
            batch_output['accuracy'] = accuracy

            for k in ['loss', 'accuracy']:
                history[k].append(batch_output[k])
                history[k] = history[k][-trail:]

            print('step = {:08}, loss = {:.3f}, accuracy-mean = {:.3f}'.format(
                step,
                np.mean(history['loss']), np.mean(history['accuracy'])))

            step += 1


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--train_file', default='./adjustment-data.jsonl', type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    options = parser.parse_args()

    if options.seed is None:
        options.seed = random.randint(0, 1e7)

    run(options)
