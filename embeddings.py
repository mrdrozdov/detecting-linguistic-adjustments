import os
import hashlib
from collections import OrderedDict

import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
from tqdm import tqdm


def validate_word2idx(word2idx):
    vocab = [w for w, i in sorted(word2idx.items(), key=lambda x: x[1])]
    for i, w in enumerate(vocab):
        assert word2idx[w] == i


def hash_vocab(vocab, version='v1.0.0'):
    m = hashlib.sha256()
    m.update(str.encode(version))
    for w in sorted(vocab):
        m.update(str.encode(w))
    return m.hexdigest()


def save_elmo_cache(path, vectors):
    np.save(path, vectors)


def load_elmo_cache(path, order):
    vectors = np.load(path)
    assert len(order) == len(vectors)
    return vectors[order]


def context_insensitive_character_embeddings(weights_path, options_path, word2idx, cuda=False, cache_dir=None):
    """
    Embeddings are always saved in sorted order (by vocab) and loaded according to word2idx.
    """
    validate_word2idx(word2idx)

    vocab = list(sorted(word2idx.keys()))
    sorted_word2idx = {k: i for i, k in enumerate(vocab)}
    order = [sorted_word2idx[w] for w, i in sorted(word2idx.items(), key=lambda x: x[1])]

    if cache_dir is not None:
        key = hash_vocab(vocab)
        cache_path = os.path.join(cache_dir, 'elmo_{}.npy'.format(key))

        if os.path.exists(cache_path):
            print('Loading cached elmo vectors: {}'.format(cache_path))
            return load_elmo_cache(cache_path, order)

    if cuda:
        device = 0
    else:
        device = -1

    batch_size = 256
    nbatches = len(vocab) // batch_size + 1

    # TODO: Does not support padding.
    elmo = ElmoEmbedder(options_file=options_path, weight_file=weights_path, cuda_device=device)
    vec_lst = []
    for i in tqdm(range(nbatches), desc='elmo'):
        start = i * batch_size
        batch = vocab[start:start+batch_size]
        if len(batch) == 0:
            continue
        vec = elmo.embed_sentence(batch)
        vec_lst.append(vec)

    vectors = np.concatenate([x[0] for x in vec_lst], axis=0)

    vectors[word2idx['_PAD']] = 0
    vectors[word2idx['[SEP]']] = np.random.randn(vectors.shape[1])
    vectors[word2idx['[CLS]']] = np.random.randn(vectors.shape[1])

    if cache_dir is not None:
        print('Saving cached elmo vectors: {}'.format(cache_path))
        save_elmo_cache(cache_path, vectors)

    return vectors[order]
