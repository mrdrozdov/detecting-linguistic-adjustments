import json
import torch
import torch.utils.data
import numpy as np


class AdjustmentDatasetBaseline(object):
    EOS_TOKEN = '[SEP]'
    TASK_TOKEN = '[CLS]'
    PADDING_TOKEN = "_PAD"
    UNK_TOKEN = "_"
    EXISTING_VOCAB_TOKEN = "unused-token-a7g39i"

    def __init__(self, use_cls_token=True, use_sep_token=True, lowercase=True):
        super(AdjustmentDatasetBaseline, self).__init__()
        self.use_cls_token = use_cls_token
        self.use_sep_token = use_sep_token
        self.lowercase = lowercase

    def read(self, path):
        sentences = []
        extra = {}
        metadata = {}

        with open(path) as f:
            for line in f:
                ex = json.loads(line)
                if self.lowercase:
                    ex['original'] = [x.lower() for x in ex['original']]
                    ex['modified'] = [x.lower() for x in ex['modified']]
                s = []
                if self.use_cls_token:
                    s += [self.TASK_TOKEN]
                s += ex['original']
                if self.use_sep_token:
                    s += [self.EOS_TOKEN]
                s += ex['modified']
                if self.use_sep_token:
                    s += [self.EOS_TOKEN]
                sentences.append(s)
                extra.setdefault('example_ids', []).append(ex['example_id'])
                extra.setdefault('labels', []).append(ex['adjustment_label'])

        def indexify_tokens(sentences):
            word2idx = {}
            word2idx[self.PADDING_TOKEN] = len(word2idx)
            word2idx[self.TASK_TOKEN] = len(word2idx)
            word2idx[self.EOS_TOKEN] = len(word2idx)
            word2idx[self.UNK_TOKEN] = len(word2idx)
            word2idx[self.EXISTING_VOCAB_TOKEN] = len(word2idx)
            def helper():
                for s in sentences:
                    for x in s:
                        if x not in word2idx:
                            word2idx[x] = len(word2idx)
                for s in sentences:
                    yield [word2idx[x] for x in s]
            sentences = list(helper())
            return sentences, word2idx
        sentences, metadata['word2idx'] = indexify_tokens(sentences)

        def indexify_labels(labels):
            def helper():
                label2idx = {}
                for x in labels:
                    if x not in label2idx:
                        label2idx[x] = len(label2idx)
                    yield label2idx[x]
            return list(helper())
        extra['labels'] = indexify_labels(extra['labels'])

        return {
            "sentences": sentences,
            "extra": extra,
            "metadata": metadata
        }


class AdjustmentDataset(object):
    EOS_TOKEN = '[SEP]'
    TASK_TOKEN = '[CLS]'

    def __init__(self):
        super(AdjustmentDataset, self).__init__()

    def read(self, path):
        sentences = []
        extra = {}

        with open(path) as f:
            for line in f:
                ex = json.loads(line)

                sentences.append([self.TASK_TOKEN] + ex['original'] + [self.EOS_TOKEN] + ex['modified'] + [self.EOS_TOKEN])
                extra.setdefault('example_ids', []).append(ex['example_id'])
                extra.setdefault('labels', []).append(ex['adjustment_label'])

        def indexify_labels(labels):
            def helper():
                label2idx = {}
                for x in labels:
                    if x not in label2idx:
                        label2idx[x] = len(label2idx)
                    yield label2idx[x]
            return list(helper())
        extra['labels'] = indexify_labels(extra['labels'])

        return {
            "sentences": sentences,
            "extra": extra
        }


class BatchSampler(torch.utils.data.Sampler):

    def __init__(self, data_source, batch_size, include_partial=False, rng=None, maxlen=0):
        self.data_source = data_source
        if rng is None:
            rng = np.random.RandomState(seed=11)
        self.rng = rng
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.include_partial = include_partial

    def reset(self):
        order = [i for i in range(len(self.data_source))
                 if self.maxlen <= 0
                 or len(self.data_source.dataset[i]) > self.maxlen]
        self.rng.shuffle(order)
        self.order = order
        self.index = -1

    def get_next_batch(self):
        batch_size = self.batch_size
        # Get next offset.
        index = self.index + 1
        # Get index of items in next batch.
        batch_index = self.order[index*batch_size:(index+1)*batch_size]
        # Update offset.
        self.index = index
        # Return list of relevant items.
        return batch_index

    def __iter__(self):
        self.reset()
        for _ in range(len(self)):
            yield self.get_next_batch()

    def __len__(self):
        length = len(self.order) // self.batch_size
        if self.include_partial:
            if length * self.batch_size < len(self.order):
                length += 1
        return length


class SimpleDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        item = self.dataset[index]
        return index, item

    def __len__(self):
        return len(self.dataset)


class BatchIterator(object):
    def __init__(self, sentences, extra):
        super(BatchIterator, self).__init__()
        self.sentences = sentences
        self.extra = extra
        self.loader = None

    def get_iterator(self, seed=11, num_workers=0, batch_size=4, maxlen=0, include_partial=False):

        def collate_fn(batch):
            index, sentences = zip(*batch)

            batch_map = {}
            batch_map['index'] = index
            batch_map['sentences'] = sentences

            for k, v in self.extra.items():
                batch_map[k] = [v[idx] for idx in index]

            return batch_map
        
        if self.loader is None:
            rng = np.random.RandomState(seed=seed)
            dataset = SimpleDataset(self.sentences)
            sampler = BatchSampler(dataset, batch_size=batch_size, rng=rng, maxlen=maxlen, include_partial=include_partial)
            loader = torch.utils.data.DataLoader(dataset, shuffle=(sampler is None), num_workers=num_workers, batch_sampler=sampler, collate_fn=collate_fn)
            self.loader = loader

        def myiterator():
            for batch in self.loader:
                batch_map = {}
                batch_map['sentences'] = batch['sentences']

                for k in self.extra.keys():
                    batch_map[k] = batch[k]

                yield batch_map

        return myiterator()
