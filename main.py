import argparse
import os
import json

import threading
from queue import Queue

from tqdm import tqdm

import numpy as np

import nltk

import editdistance

from sklearn.feature_extraction.text import TfidfVectorizer
from similarity.longest_common_subsequence import LongestCommonSubsequence

import faiss
from faiss import normalize_L2


def readfile(path):
    def func():
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                # Note: Only looks at first sentence.
                tree = nltk.Tree.fromstring(data['sentence1_parse'])
                pos_tuples = tree.pos()
                words = [x[0].lower() for x in tree.pos()]
                pos = [x[1] for x in tree.pos()]

                rec = {}
                rec['sentence'] = words
                rec['pos'] = pos
                yield rec
    return list(func())


class Index(object):
    def __init__(self, cuda=False, index='l2', dim=None):
        super(Index, self).__init__()
        self.cuda = cuda
        self.D, self.I = None, None

        if index == 'l2':
            index = faiss.IndexFlatL2(dim)
        elif index == 'ip':
            index = faiss.IndexFlatIP(dim)
        if cuda:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        self.index = index

    def add(self, vecs):
        self.index.add(vecs)

    def cache(self, vecs, k):
        self.D, self.I = self.index.search(vecs, k)

    def topk(self, q, k):
        for j in range(k):
            idx = self.I[q][j]
            dist = self.D[q][j]
            yield idx, dist


def run_lcs(options):
    """
    TODO: LCS is slow. First rank according to TFIDF or LEV.
    """
    records = readfile(options.data)
    sentences_tokenized = [rec['sentence'] for rec in records]
    sentences = [' '.join(x) for x in sentences_tokenized]

    n = len(sentences)

    distance_matrix = np.zeros((n, n))

    def do_work(item):
        i, j = item
        lcs = LongestCommonSubsequence()
        distance_matrix[i, j] = editdistance.eval(sentences_tokenized[i], sentences_tokenized[j])

    # The worker thread pulls an item from the queue and processes it
    def worker():
        while True:
            item = q.get()
            do_work(item)
            q.task_done()

    # Create the queue and thread pool.
    q = Queue()
    for i in range(options.n_threads):
         t = threading.Thread(target=worker)
         t.daemon = True  # thread dies when main thread (only non-daemon thread) exits.
         t.start()

    for i in tqdm(range(n)):
        for j in range(n):
            if i == j:
                continue
            q.put((i, j))

    q.join()

    n_topk = 10
    n_tosearch = 100
    topk_candidates = np.argsort(distance_matrix, axis=1)[:, :n_tosearch]

    for i in range(n):
        i_topk_candidates = topk_candidates[i].tolist()
        topk = []

        for j in i_topk_candidates:
            if i == j:
                continue
            if sentences[i] == sentences[j]:
                continue
            topk.append(j)
            if len(topk) == n_topk:
                break

        assert len(topk) == n_topk, "Did not find enough eligible candidates."

        print('i={} s={}'.format(i, sentences[i]))
        for rank, j in enumerate(topk):
            lcs_dist = distance_matrix[i, j]
            lev_dist = editdistance.eval(sentences_tokenized[i], sentences_tokenized[j])
            print('rank={} j={} lev-dist={} lcs-dist={} s={}'.format(
                rank, j, lev_dist, lcs_dist, sentences[j]))
        print()


def run_lev(options):
    records = readfile(options.data)
    sentences_tokenized = [rec['sentence'] for rec in records]
    sentences = [' '.join(x) for x in sentences_tokenized]

    n = len(sentences)

    distance_matrix = np.zeros((n, n))

    def do_work(item):
        i, j = item
        distance_matrix[i, j] = editdistance.eval(sentences_tokenized[i], sentences_tokenized[j])

    # The worker thread pulls an item from the queue and processes it
    def worker():
        while True:
            item = q.get()
            do_work(item)
            q.task_done()

    # Create the queue and thread pool.
    q = Queue()
    for i in range(options.n_threads):
         t = threading.Thread(target=worker)
         t.daemon = True  # thread dies when main thread (only non-daemon thread) exits.
         t.start()

    for i in tqdm(range(n)):
        for j in range(n):
            if i == j:
                continue
            q.put((i, j))

    q.join()

    n_topk = 10
    n_tosearch = 100
    topk_candidates = np.argsort(distance_matrix, axis=1)[:, :n_tosearch]

    # lcs
    lcs = LongestCommonSubsequence()

    for i in range(n):
        i_topk_candidates = topk_candidates[i].tolist()
        topk = []

        for j in i_topk_candidates:
            if i == j:
                continue
            if sentences[i] == sentences[j]:
                continue
            topk.append(j)
            if len(topk) == n_topk:
                break

        assert len(topk) == n_topk, "Did not find enough eligible candidates."

        print('i={} s={}'.format(i, sentences[i]))
        for rank, j in enumerate(topk):
            lev_dist = distance_matrix[i, j]
            lcs_dist = lcs.distance(sentences_tokenized[i], sentences_tokenized[j])
            print('rank={} j={} lev-dist={} lcs-dist={} s={}'.format(
                rank, j, lev_dist, lcs_dist, sentences[j]))
        print()


def run_tfidf(options):
    records = readfile(options.data)
    sentences_tokenized = [rec['sentence'] for rec in records]
    sentences = [' '.join(x) for x in sentences_tokenized]

    n = len(sentences)

    distance_matrix = np.zeros((n, n))

    max_features = None  # Note: Vocab is not limited.
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf = vectorizer.fit_transform(sentences).toarray().astype(np.float32)

    n_topk = 10
    n_tosearch = 100
    normalize_L2(tfidf)
    index = Index(index='ip', dim=tfidf.shape[1])
    index.add(tfidf)
    index.cache(tfidf, n_tosearch)

    # lcs
    lcs = LongestCommonSubsequence()

    for i in range(n):
        topk_candidates = list(index.topk(i, n_tosearch))
        topk = []

        for item in topk_candidates:
            j, _ = item
            if i == j:
                continue
            if sentences[i] == sentences[j]:
                continue
            topk.append(item)
            if len(topk) == n_topk:
                break

        assert len(topk) == n_topk, "Did not find enough eligible candidates."

        print('i={} s={}'.format(i, sentences[i]))
        for rank, item in enumerate(topk):
            j, tfidf_dist = item
            lev_dist = editdistance.eval(sentences_tokenized[i], sentences_tokenized[j])
            lcs_dist = lcs.distance(sentences_tokenized[i], sentences_tokenized[j])
            print('rank={} j={} tfidf-dist={} lev-dist={} lcs-dist={} s={}'.format(
                rank, j, tfidf_dist, lev_dist, lcs_dist, sentences[j]))
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=os.path.expanduser('~/data/ptb.jsonl'), type=str,
        help='Path to file formatted in style of SNLI.')
    parser.add_argument('--mode', default='lcs', choices=('lcs', 'lev', 'tfidf'))
    parser.add_argument('--n_threads', default=2, type=int)
    options = parser.parse_args()

    if options.mode == 'lcs':
        run_lcs(options)
    if options.mode == 'lev':
        run_lev(options)
    if options.mode == 'tfidf':
        run_tfidf(options)
