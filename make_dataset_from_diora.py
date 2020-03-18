import json
import random
import os
import sys

import torch
import torch.nn as nn

import nltk
import numpy as np
from tqdm import tqdm

from diora import build_net
from diora_utils import get_offset_lookup
from embeddings import context_insensitive_character_embeddings

HOME = os.path.expanduser('~')


def nltk_tree_to_spans(tr):
    spans = []

    def helper(tr, pos=0, depth=0):
        if len(tr) == 1 and isinstance(tr[0], str):
            size = 1
            label = tr.label()
            #UNCOMMENT to include part-of-speech.
            #spans.append((pos, size, label, depth))
            return size
        if len(tr) == 1:
            size = helper(tr[0], pos, depth + 1)
            label = tr.label()
            spans.append((pos, size, label, depth))
            return size
        size = 0
        for x in tr:
            xsize = helper(x, pos + size, depth + 1)
            size += xsize
        label = tr.label()
        spans.append((pos, size, label, depth))
        return size

    helper(tr)

    return spans


class ReadNLI(object):
    SENTENCE_KEYS = ['sentence1', 'sentence2']
    TREE_KEYS = ['sentence1_parse', 'sentence2_parse']

    def __init__(self, lowercase=True, max_length=0):
        self.lowercase = lowercase
        self.max_length = max_length

    def read(self, path):
        data = {'example_id': []}
        metadata = {}

        with open(path) as f:
            for line in tqdm(f, desc='read'):
                skip = False

                ex = json.loads(line)

                for k in self.SENTENCE_KEYS:
                    v = ex[k].strip().split()
                    if self.lowercase:
                        v = [w.lower() for w in v]
                    if self.max_length > 0 and len(v) > self.max_length:
                        skip = True

                if skip:
                    continue

                for k in self.TREE_KEYS:
                    v = ex[k].strip()
                    tr = nltk.Tree.fromstring(v)
                    spans = nltk_tree_to_spans(tr)
                    ex[k + '_spans'] = spans

                for k, v in ex.items():
                    data.setdefault(k, []).append(v)

                if 'example_id' not in ex:
                    example_id = len(data['example_id'])
                    data['example_id'].append(example_id)

        check_length = None
        for k in data.keys():
            if check_length is None:
                check_length = len(data[k])
            assert check_length == len(data[k]), 'All columns have the same length.'

        dataset = {}
        dataset['data'] = data
        dataset['metadata'] = metadata
        return dataset


class Vocab(object):
    PADDING_TOKEN = "_PAD_"
    UNK_TOKEN = "_UNK_"
    NUM_RESERVED_TOKENS = 100

    def __init__(self):
        self.frozen = False
        self.word2idx = None

    def init(self):
        word2idx = {}

        special_tokens = [self.PADDING_TOKEN, self.UNK_TOKEN]
        for tok in special_tokens:
            word2idx[tok] = len(word2idx)

        for i in range(self.NUM_RESERVED_TOKENS):
            tok = '_RESERVED_{}_'.format(i)
            word2idx[tok] = len(word2idx)

        self.word2idx = word2idx

    def add(self, sentences):
        assert self.frozen == False, "Can not add sentence if frozen."
        for s in sentences:
            for w in s:
                if w not in self.word2idx:
                    self.word2idx[w] = len(self.word2idx)

    def freeze(self):
        self.frozen = True

    def indexify(self, sentences):
        assert self.frozen == True, "Must freeze before indexifying."
        new_sentences = []
        for s in sentences:
            new_sentences.append([self.word2idx[w] for w in s])
        return new_sentences


def get_cells_for_spans(chart, spans, example_id):
    batch_size, length, cell_size = chart['info']['batch_size'], chart['info']['length'], chart['info']['size']
    offset_lookup = get_offset_lookup(length)
    batch_index, cell_index, span_info = [], [], []

    for i in range(batch_size):
        for sp in spans[i]:
            pos, size, label, depth = sp
            local_span_info = {}
            local_span_info['position'] = pos
            local_span_info['size'] = size
            local_span_info['label'] = label
            local_span_info['depth'] = depth
            local_span_info['example_id'] = example_id[i]
            span_info.append(local_span_info)

            level = size - 1
            batch_index.append(i)
            cell_index.append(offset_lookup[level] + pos)

    inside = chart['inside_h'][batch_index, 0, cell_index]
    outside = chart['outside_h'][batch_index, 0, cell_index]
    cell = torch.cat([inside, outside], dim=1)

    assert cell.shape == (len(cell_index), 2 * cell_size)

    return span_info, cell


class PhraseData(object):
    def __init__(self):
        self.frozen = False
        self.data = None

    def init(self):
        self.data = {}

    def add(self, span_info, cell):
        assert self.frozen == False
        self.data.setdefault('span_info', []).append(span_info)
        self.data.setdefault('cell', []).append(cell.data.cpu().numpy())

    def freeze(self):
        self.frozen = True

        span_info = []
        for lst in self.data['span_info']:
            for x in lst:
                span_info.append(x)
        self.data['span_info'] = span_info

        self.data['cell'] = np.concatenate(self.data['cell'], axis=0)

    def write(self, fn_info, fn_vec):
        with open(fn_info, 'w') as f:
            for x in self.data['span_info']:
                f.write(json.dumps(x, sort_keys=True) + '\n')
        np.savetxt(fn_vec, self.data['cell'])


def run(options):
    print('FLAGS:')
    print(json.dumps(options.__dict__, sort_keys=True))

    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)

    # Read data.
    dataset = ReadNLI(lowercase=True, max_length=options.max_length).read(options.file_in)

    vocab = Vocab()
    vocab.init()
    vocab.add(dataset['data']['sentence1'])
    vocab.add(dataset['data']['sentence2'])
    vocab.freeze()
    word2idx = vocab.word2idx

    dataset['data']['sentence1'] = vocab.indexify(dataset['data']['sentence1'])
    dataset['data']['sentence2'] = vocab.indexify(dataset['data']['sentence2'])

    embeddings = context_insensitive_character_embeddings(
        options.elmo_weight_file,
        options.elmo_options_file,
        word2idx=word2idx,
        cuda=options.cuda,
        cache_dir=options.cache,
        zero_tokens=[Vocab.PADDING_TOKEN] + list(['_RESERVED_{}_'.format(i) for i in range(Vocab.NUM_RESERVED_TOKENS)]),
        rand_tokens=[Vocab.UNK_TOKEN],
        )

    net = build_net(options, embeddings)
    if options.cuda:
        net.cuda()

    def pipeline(batch_map):
        batch_map['sentence1'] = torch.LongTensor(batch_map['sentence1'])
        if options.cuda:
            batch_map['sentence1'] = batch_map['sentence1'].cuda()
        return batch_map

    class Sampler(object):
        def __init__(self, data, example_id):
            self.data = data
            self.example_id = example_id
            assert len(data) == len(example_id)

        def init(self):
            buckets = {}
            for idx, s in zip(self.example_id, self.data):
                buckets.setdefault(len(s), []).append(idx)

            self.current_bucket = 0
            self.bucket_order = list(sorted(buckets.keys()))
            self.buckets = buckets
            self.bucket_offset = {k: 0 for k in buckets.keys()}

        def sample_sentence_ids(self, n=4):
            sentence_ids = []

            while True:
                bucket_idx = self.current_bucket
                if bucket_idx == len(self.bucket_order):
                    return None

                bucket_k = self.bucket_order[bucket_idx]
                bucket = self.buckets[bucket_k]
                bucket_offset = self.bucket_offset[bucket_k]
                bucket_size = len(bucket)
                if bucket_offset == bucket_size:
                    self.current_bucket += 1
                    if len(sentence_ids) > 0:
                        break
                    continue

                self.bucket_offset[bucket_k] += 1

                sentence_ids.append(bucket[bucket_offset])
                if len(sentence_ids) == n:
                    break

            return sentence_ids

    # One sentence at a time.
    sampler = Sampler(dataset['data']['sentence1'], dataset['data']['example_id'])
    sampler.init()
    phrase_data = PhraseData()
    phrase_data.init()

    with torch.no_grad():
        for i in tqdm(range(options.num_iter), desc='encode'):

            sentence_ids = sampler.sample_sentence_ids(n=options.batch_size)
            if sentence_ids is None:
                print('Ending early. Completed {}/{}'.format(i, options.num_iter))
                break

            batch_map = {}
            batch_map['sentence_id'] = sentence_ids
            for k in dataset['data'].keys():
                batch_map[k] = [dataset['data'][k][idx] for idx in batch_map['sentence_id']]
            batch_map = pipeline(batch_map)

            out = net(batch_map['sentence1'])
            span_info, cell = get_cells_for_spans(out['chart'], batch_map['sentence1_parse_spans'], batch_map['example_id'])

            phrase_data.add(span_info, cell)

    phrase_data.freeze()

    phrase_data.write(options.file_out_span_info, options.file_out_span_vec)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=11, type=int)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--size', default=400, type=int)
    parser.add_argument('--num_iter', default=100, type=int)
    parser.add_argument('--max_length', default=40, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--file_in', default=os.path.join(HOME, 'data/snli_1.0/snli_1.0_dev.jsonl'), type=str)
    parser.add_argument('--file_out_span_vec', default='./span-vec.npy', type=str)
    parser.add_argument('--file_out_span_info', default='./span-info.jsonl', type=str)
    parser.add_argument('--cache', default='./cache', type=str)
    parser.add_argument('--diora_file', default='http://diora-naacl-2019.s3.amazonaws.com/diora-checkpoints.zip', type=str)
    parser.add_argument('--elmo_options_file', default='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json', type=str)
    parser.add_argument('--elmo_weight_file',  default='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5', type=str)
    options = parser.parse_args()

    if options.seed is None:
        options.seed = random.randint(0, 1e7)

    run(options)
