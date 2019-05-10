"""
Creates a dataset of:
- same cat. swaps
- diff. cat. swaps
- non-constituent swaps
- nonsense swaps
"""

import argparse
import itertools
import json
import os
import random
import re

import nltk


def add_spaced(string):
    string = string.replace("(", "( ")
    string = string.replace(")", " )")
    return string


def binary_parse_to_tokens(parse):
    parse_tokens = add_spaced(parse).split()
    return [x for x in parse_tokens if x not in ('(', ')')]


def preprocess_parse(parse):
    parse_split = re.split(r'[()]', parse)
    parse_filter_0 = filter(lambda x: x == None or x != '', parse_split)
    parse_filter_1 = filter(lambda x: x != ' ', parse_filter_0)
    return list(parse_filter_1)


def parse_to_indexed_contituents_labeled(parse):
    # Parse the parse.
    parse_tokens = add_spaced(parse).split()
    listo_parse = preprocess_parse(parse)

    # If only one word, exit early.
    if len(parse_tokens) == 1:
        return [(0, 1)]

    result = {}
    backpointers = []
    indexed_constituents = []
    word_index = 0

    # First, get all constituents.
    for index, token in enumerate(parse_tokens):
        if token == '(':
            backpointers.append(word_index)
        elif token == ')':
            start = backpointers.pop()
            end = word_index
            constituent = (start, end)
            indexed_constituents.append(constituent)
        else:
            word_index += 1

    indexed_constituents = sorted(indexed_constituents)

    # Next, create a dictionary label->spans.
    for i in range(len(indexed_constituents)):
        label = listo_parse[i].split()[0] # NP, VP, etc.
        span = indexed_constituents[i] # (start, end)

        result.setdefault(label, []).append(span)
        # The line above is the same as the following:
        # if label not in result:
        #     result[label] = []
        # result[label].append(span)

    return result


class Node(object):
    def __init__(self, parse, label):
        super(Node, self).__init__()
        self.parse = parse
        self.label = label

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '({}, {})'.format(self.parse, self.label)


def parse_to_tuples(parse, label=None):
    if isinstance(parse, str):
        return Node(parse, label)
    result = tuple(parse_to_tuples(x, parse.label()) for x in parse)
    while len(result) == 1: # unary-chains are dropped
        return result[0]
    return Node(result, parse.label())


def nodes_to_labeled_spans(node, ignore_pos=False):
    result = []

    def helper(node, pos=0):
        if isinstance(node.parse, str):
            if not ignore_pos:
                result.append((pos, 1, node.label))
            return 1

        sofar = 0
        for x in node.parse:
            xsize = helper(x, pos + sofar)
            sofar += xsize

        size = sofar
        result.append((pos, size, node.label))

        return size

    helper(node)

    return result


def parse_to_indexed_contituents_labeled_v2(parse):
    nodes = parse_to_tuples(nltk.Tree.fromstring(parse))
    labeled_spans = nodes_to_labeled_spans(nodes, ignore_pos=False)

    result = {}
    for pos, size, label in labeled_spans:
        result.setdefault(label, []).append((pos, pos+size))
    return result


def print_labeled_spans(label2spans, tokens):
    for label, lst in label2spans.items():
        for start, end in lst:
            phrase = ' '.join(tokens[start:end])
            print('\t{}\t{}\t{}'.format(label, (start, end), phrase))


def swap_phrase(tokens, replacement, span):
    parts = []

    # Left of replacement.
    if span[0] > 0:
        parts.append(' '.join(tokens[:span[0]]))

    # Replacement.
    parts.append(replacement)

    # Right of replacement.
    if span[1] < len(tokens):
        parts.append(' '.join(tokens[span[1]:]))

    new_sentence = ' '.join(parts)

    return new_sentence


def run(options):
    print('seed', options.seed)

    random.seed(options.seed)

    print('reading')

    dataset = []

    with open(options.file_in) as f:
        for line in f:
            ex = json.loads(line)
            example_id = ex['pairID']
            parse = ex['sentence1_parse']
            binary_parse = ex['sentence1_binary_parse']
            tokens = binary_parse_to_tokens(binary_parse)
            label2spans = parse_to_indexed_contituents_labeled_v2(parse)

            datapoint = {}
            datapoint['example_id'] = example_id
            datapoint['tokens'] = tokens
            datapoint['label2spans'] = label2spans

            dataset.append(datapoint)

            # Note: I think this does not necessarily account for constituents correctly.
            # constituents = parse_to_indexed_contituents_labeled(parse)
            # print(example_id)
            # print_labeled_spans(constituents, tokens)

    # Book-keeping
    all_valid_constituents = set()
    vocab = set()

    print('book-keeping')

    for x in dataset:
        tokens = x['tokens']
        label2spans = x['label2spans']
        vocab.update(tokens)
        for label, lst in label2spans.items():
            for start, end in lst:
                phrase = ' '.join(tokens[start:end])
                all_valid_constituents.add(phrase)

    print('creating dictionary label->phrase for all phrases')

    label2phrase = {}

    for x in dataset:
        tokens = x['tokens']
        label2spans = x['label2spans']

        # Book-keeping.
        vocab.update(tokens)

        for label, lst in label2spans.items():
            for start, end in lst:
                phrase = ' '.join(tokens[start:end])
                label2phrase.setdefault(label, set()).add(phrase)

                # Book-keeping.
                all_valid_constituents.add(phrase)

    # Convert to lists instead of sets.
    label2phrase = {k: list(v) for k, v in label2phrase.items()}

    print('creating collection of nonconstituent phrases')

    example2nonconstituents = {}

    for i, x in enumerate(dataset):
        tokens = x['tokens']
        label2spans = x['label2spans']
        span_set = set(itertools.chain(*label2spans.values()))
        length = len(tokens)
        max_attempts = length
        nonconstituents = set()
        available_sizes = [x[1]-x[0] for x in span_set if x[1]-x[0] < length]

        if len(available_sizes) == 0:
            print('skip', i, tokens)
            continue

        for _ in range(max_attempts):
            size = random.choice(available_sizes)
            pos = random.choice(range(0, length-size))
            new_span = (pos, pos+size)
            if new_span not in span_set:
                phrase = ' '.join(tokens[new_span[0]:new_span[1]])
                if phrase not in all_valid_constituents:
                    nonconstituents.add(phrase)

        if len(nonconstituents) > 0:
            example2nonconstituents[i] = list(nonconstituents)
        else:
            print('skip', i, tokens)

    print('creating collection of nonsense')

    example2nonsense = {}

    for i, x in enumerate(dataset):
        tokens = x['tokens']
        label2spans = x['label2spans']
        span_set = set(itertools.chain(*label2spans.values()))
        length = len(tokens)
        max_attempts = length
        nonsense = set()
        available_sizes = [x[1]-x[0] for x in span_set]

        if len(available_sizes) == 0:
            print('skip', i, tokens)
            continue

        for _ in range(max_attempts):
            size = random.choice(available_sizes)
            phrase = ' '.join(random.sample(vocab, size))
            if phrase not in all_valid_constituents:
                nonsense.add(phrase)

        if len(nonsense) > 0:
            example2nonsense[i] = list(nonsense)
        else:
            print('skip', i, tokens)

    print('create random sentences')

    with open(options.file_out, 'w') as f:
        for x in dataset:
            example_id = x['example_id']
            tokens = x['tokens']
            label2spans = x['label2spans']

            for i in range(options.n_swaps_per_sentence):
                # First choose a label.
                label = random.choice(list(label2spans.keys()))
                # Then choose a span.
                span = random.choice(label2spans[label])

                phrase = ' '.join(tokens[span[0]:span[1]])

                for j in range(options.n_candidates_per_swap):
                    # Same Cat
                    replacement = random.choice(label2phrase[label])
                    new_sentence = swap_phrase(tokens, replacement, span).split(' ')

                    ex = { "example_id": example_id, "original": tokens, "modified": new_sentence, "span_label": label, "span": span, "adjustment_label": "same-cat" }
                    f.write('{}\n'.format(json.dumps(ex)))

                    # Diff Cat
                    new_label = random.choice(list(label2phrase.keys()))
                    replacement = random.choice(label2phrase[new_label])
                    new_sentence = swap_phrase(tokens, replacement, span).split(' ')

                    ex = { "example_id": example_id, "original": tokens, "modified": new_sentence, "span_label": label, "target_label": new_label, "span": span, "adjustment_label": "diff-cat" }
                    f.write('{}\n'.format(json.dumps(ex)))

                    # Non-constituent
                    _example_id = random.sample(example2nonconstituents.keys(), 1)[0]
                    replacement = random.choice(example2nonconstituents[_example_id])
                    new_sentence = swap_phrase(tokens, replacement, span).split(' ')

                    ex = { "example_id": example_id, "original": tokens, "modified": new_sentence, "span_label": label, "span": span, "adjustment_label": "non-constituent" }
                    f.write('{}\n'.format(json.dumps(ex)))

                    # Nonsense
                    _example_id = random.sample(example2nonsense.keys(), 1)[0]
                    replacement = random.choice(example2nonsense[_example_id])
                    new_sentence = swap_phrase(tokens, replacement, span).split(' ')

                    ex = { "example_id": example_id, "original": tokens, "modified": new_sentence, "span_label": label, "span": span, "adjustment_label": "nonsense" }
                    f.write('{}\n'.format(json.dumps(ex)))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--n_swaps_per_sentence', default=2, type=int)
    parser.add_argument('--n_candidates_per_swap', default=5, type=int)
    parser.add_argument('--random_label', action='store_true')
    parser.add_argument('--file_in', default=os.path.expanduser('~/data/ptb-dev.jsonl'), type=str)
    parser.add_argument('--file_out', default='adjustment-data.jsonl', type=str)
    options = parser.parse_args()

    if options.seed is None:
        options.seed = random.randint(0, 1e7)

    run(options)
