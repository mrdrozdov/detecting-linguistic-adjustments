import os
import json
import random

from read_data import *


def run(options):
    num_epochs = 10

    train_dataset = AdjustmentDataset().read(options.train_file)
    train_iterator = BatchIterator(train_dataset['sentences'], train_dataset['extra'])

    for epoch in range(num_epochs):
        for batch_map in train_iterator.get_iterator():
            batch_size = len(batch_map['sentences'])
            max_length = max([len(x) for x in batch_map['sentences']])
            print('batch-size={} length={}'.format(batch_size, max_length))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--train_file', default='./adjustment-data.jsonl', type=str)
    options = parser.parse_args()

    if options.seed is None:
        options.seed = random.randint(0, 1e7)

    run(options)
