"""
think we can use this function to
(1) randomly choose a sentence,
(2) get the keys from its dict,
(3) pick a random key from its keys,
(4) randomly pick one of that key’s values,
(5) get the span corresponding to it, then
(6) find a sentence with the same key and replace one of its values with the span."
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

    print('creating dictionary label->phrase for all phrases')

    label2phrase = {}

    for x in dataset:
        tokens = x['tokens']
        label2spans = x['label2spans']

        for label, lst in label2spans.items():
            for start, end in lst:
                phrase = ' '.join(tokens[start:end])
                label2phrase.setdefault(label, set()).add(phrase)

    # Convert to lists instead of sets.
    label2phrase = {k: list(v) for k, v in label2phrase.items()}

    print('create random sentences')

    for x in dataset:
        example_id = x['example_id']
        tokens = x['tokens']
        label2spans = x['label2spans']

        print('example_id={} length={} sentennce={}'.format(example_id, len(tokens), ' '.join(tokens)))

        for i in range(options.n_swaps_per_sentence):
            # First choose a label.
            label = random.choice(list(label2spans.keys()))
            # Then choose a span.
            span = random.choice(label2spans[label])

            phrase = ' '.join(tokens[span[0]:span[1]])

            print('{}. {} {}'.format(i, label, phrase))

            for j in range(options.n_candidates_per_swap):
                # Then choose a replacement.
                replacement = random.choice(label2phrase[label])

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

                print('\t{}. {}'.format(j, new_sentence))

        print()
        


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--n_swaps_per_sentence', default=2, type=int)
    parser.add_argument('--n_candidates_per_swap', default=5, type=int)
    parser.add_argument('--file_in', default=os.path.expanduser('~/data/ptb-dev.jsonl'), type=str)
    options = parser.parse_args()

    if options.seed is None:
        options.seed = random.randint(0, 1e7)

    run(options)
