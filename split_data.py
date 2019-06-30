import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--nva', default=1000, type=int)
parser.add_argument('--nte', default=1000, type=int)
parser.add_argument('--ntr', default=None, type=int)
parser.add_argument('--seed', default=1331, type=int)
parser.add_argument('--file_in', default='adjustment-data.jsonl', type=str)
parser.add_argument('--file_tr', default='adjustment-tr.jsonl', type=str)
parser.add_argument('--file_va', default='adjustment-va.jsonl', type=str)
parser.add_argument('--file_te', default='adjustment-te.jsonl', type=str)
options = parser.parse_args()

M = 4

data = {}
with open(options.file_in) as f:
    for i, line in enumerate(f):
        data.setdefault(i // M, []).append(line.rstrip())

for k, v in data.items():
    assert len(v) == M

N = len(data)
index = list(range(N))
random.seed(options.seed)
random.shuffle(index)

sofar = 0

touse = options.nva
va = index[sofar:sofar+touse]
sofar += touse

touse = options.nte
te = index[sofar:sofar+touse]
sofar += touse

if options.ntr is not None:
    touse = options.ntr
    tr = index[sofar:sofar+touse]
else:
    tr = index[sofar:]

def write_file(data, index, filename):
    with open(filename, 'w') as f:
        for i in index:
            for line in data[i]:
                f.write(line + '\n')

write_file(data, va, options.file_va)
write_file(data, te, options.file_te)
write_file(data, tr, options.file_tr)

