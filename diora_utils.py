import collections

import torch


def build_chart(batch_size, length, size, repeat=1, init={}, dtype=torch.float, device=None):
    ncells = int(length * (1 + length) / 2)

    chart = {}
    chart['inside_h'] = torch.full((batch_size, repeat, ncells, size), 0, dtype=dtype, device=device)
    chart['inside_s'] = torch.full((batch_size, repeat, ncells, 1), 0, dtype=dtype, device=device)
    chart['outside_h'] = torch.full((batch_size, repeat, ncells, size), 0, dtype=dtype, device=device)
    chart['outside_s'] = torch.full((batch_size, repeat, ncells, 1), 0, dtype=dtype, device=device)

    for k, v in init.items():
        name, i_repeat, i_cell_start, i_cell_end = k
        if i_cell_end is not None:
            chart[name][:, i_repeat, i_cell_start:i_cell_end] = v
        else:
            chart[name][:, i_repeat, i_cell_start:] = v

    chart['info'] = {}
    chart['info']['batch_size'] = batch_size
    chart['info']['length'] = length
    chart['info']['size'] = size
    chart['info']['device'] = device

    return chart


def get_offset_lookup(length):
    lookup = {}
    ncells = int(length * (1 + length) / 2)
    for level in range(length):
        L = length - level
        local_ncells = int(L * (1 + L) / 2)
        lookup[level] = ncells - local_ncells
    return lookup


class IndexUtil(object):
    def __init__(self, cache_on=True):
        self.cache = collections.defaultdict(dict)
        self.cache_on = cache_on

    def cache_func(self, func, name, key):
        if self.cache_on:
            self.cache[key] = func()
            return self.cache[key]
        return func()


class InsideUtil(object):
    @staticmethod
    def get_pairs(length, level, pos):
        N = level

        pairs = []
        for i_component in range(0, N):
            l_level = i_component
            l_pos = pos
            l_size = i_component + 1
            r_level = level - l_level - 1
            r_pos = l_pos + l_size

            assert l_level >= 0 and l_level < length, l_level
            assert l_pos >= 0 and l_pos < length, l_pos
            assert r_level >= 0 and r_level < length, r_level
            assert r_pos >= 0 and r_pos < length, r_pos

            pair = ((l_level, l_pos), (r_level, r_pos))
            pairs.append(pair)

        return pairs

    @staticmethod
    def get_components(length, level, offset_lookup=None):
        if offset_lookup is None:
            offset_lookup = get_offset_lookup(length)

        L = length - level
        N = level

        output = collections.defaultdict(list)
        for pos in range(L):
            x = (level, pos)
            x_idx = offset_lookup[level] + pos
            pairs = InsideUtil.get_pairs(length, level, pos)
            assert len(pairs) == N

            for i, (l, r), in enumerate(pairs):
                output['i_component'].append(i)
                output['x'].append(x)
                output['l'].append(l)
                output['r'].append(r)

                output['x_idx'].append(x_idx)

                l_level, l_pos = l
                l_idx = offset_lookup[l_level] + l_pos
                output['l_idx'].append(l_idx)

                r_level, r_pos = r
                r_idx = offset_lookup[r_level] + r_pos
                output['r_idx'].append(r_idx)

        return output


class OutsideUtil(object):
    @staticmethod
    def get_pairs(length, level, pos):
        N = length - level - 1
        size = level + 1
        num_to_left = pos

        pairs = []
        for i_component in range(0, N):
            if i_component < pos:
                # Sibling on left.
                s_pos = i_component
                s_size = pos - s_pos
                s_level = s_size - 1

                p_pos = s_pos
            else:
                # Sibling on right.
                s_pos = pos + size
                s_size = (i_component - num_to_left) + 1
                s_level = s_size - 1

                p_pos = pos

            p_size = s_size + size
            p_level = p_size - 1

            assert p_level >= 0 and p_level < length, p_level
            assert p_pos >= 0 and p_pos < length, p_pos
            assert s_level >= 0 and s_level < length, s_level
            assert s_pos >= 0 and s_pos < length, s_pos

            pair = ((p_level, p_pos), (s_level, s_pos))
            pairs.append(pair)

        return pairs

    @staticmethod
    def get_components(length, level, offset_lookup=None):
        if offset_lookup is None:
            offset_lookup = get_offset_lookup(length)

        L = length - level
        N = length - level - 1

        output = collections.defaultdict(list)
        for pos in range(L):
            x = (level, pos)
            x_idx = offset_lookup[level] + pos
            pairs = OutsideUtil.get_pairs(length, level, pos)
            assert len(pairs) == N

            for i, (p, s), in enumerate(pairs):
                output['i_component'].append(i)
                output['x'].append(x)
                output['p'].append(p)
                output['s'].append(s)

                output['x_idx'].append(x_idx)

                p_level, p_pos = p
                p_idx = offset_lookup[p_level] + p_pos
                output['p_idx'].append(p_idx)

                s_level, s_pos = s
                s_idx = offset_lookup[s_level] + s_pos
                output['s_idx'].append(s_idx)

        return output
