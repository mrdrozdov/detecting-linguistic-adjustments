import torch
import torch.nn as nn

from diora_utils import *


def build_net(options, embeddings):
    input_size = embeddings.shape[1]
    size = options.size

    embed = Embed(nn.Embedding.from_pretrained(torch.from_numpy(embeddings), freeze=True), input_size, size)
    diora = DIORA(input_size=size, size=size, k=1)
    net = Net(embed, diora)

    return net


class Net(nn.Module):
    def __init__(self, embed, diora):
        super().__init__()
        self.embed = embed
        self.diora = diora

    def forward(self, x, run_args={}):
        h = self.embed(x)
        out = self.diora(h, run_args)
        return out


class Embed(nn.Module):
    def __init__(self, embeddings, input_size, size):
        super().__init__()
        self.input_size = input_size
        self.size = size
        self.embeddings = embeddings
        self.mat = nn.Parameter(torch.FloatTensor(size, input_size))
        self.reset_parameters()

    def reset_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for i, param in enumerate(params):
            param.data.normal_()

    def forward(self, x):
        batch_size, length = x.shape
        e = self.embeddings(x.view(-1))
        t = torch.mm(e, self.mat.t()).view(batch_size, length, -1)
        return t


class UnitNorm(object):
    def __call__(self, x, p=2, eps=1e-8):
        return x / x.norm(p=p, dim=-1, keepdim=True).clamp(min=eps)


class Bilinear(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.mat = nn.Parameter(torch.FloatTensor(self.size, self.size))
        self.reset_parameters()

    def reset_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for p in params:
            p.data.normal_()

    def forward(self, h0, h1):
        z = torch.matmul(h0, self.mat).unsqueeze(-2)
        s = torch.matmul(z, h1.unsqueeze(-1))
        return s.squeeze(-1)


class ComposeMLP(nn.Module):
    def __init__(self, size, input_size=None, n_layers=2, leaf=False):
        super().__init__()

        self.size = size
        self.input_size = input_size
        self.n_layers = n_layers

        if leaf:
            self.V = nn.Parameter(torch.FloatTensor(self.input_size, self.size))
        self.W = nn.Parameter(torch.FloatTensor(2 * self.size, self.size))
        self.B = nn.Parameter(torch.FloatTensor(self.size))

        for i in range(1, n_layers):
            setattr(self, 'W_{}'.format(i), nn.Parameter(torch.FloatTensor(self.size, self.size)))
            setattr(self, 'B_{}'.format(i), nn.Parameter(torch.FloatTensor(self.size)))

        self.reset_parameters()

    def reset_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for p in params:
            p.data.normal_()

    def leaf_transform(self, x):
        return torch.tanh(torch.matmul(x, self.V) + self.B)

    def forward(self, h0, h1):
        input_h = torch.cat([h0, h1], -1)
        h = torch.relu(torch.matmul(input_h, self.W) + self.B)
        for i in range(1, self.n_layers):
            W = getattr(self, 'W_{}'.format(i))
            B = getattr(self, 'B_{}'.format(i))
            h = torch.relu(torch.matmul(h, W) + B)

        return h


class DIORA(nn.Module):
    def __init__(self, input_size, size, k=1):
        super().__init__()
        self.input_size = input_size
        self.size = size
        self.k = k

        self.inside_compose = ComposeMLP(self.size, n_layers=2, leaf=True, input_size=self.input_size)
        self.outside_compose = ComposeMLP(self.size, n_layers=2, leaf=False)

        self.inside_score = Bilinear(self.size)
        self.outside_score = Bilinear(self.size)

        self.mat = nn.Parameter(torch.FloatTensor(size, size))

        self.outside_h_root = nn.Parameter(torch.FloatTensor(size))

        self.iu = IndexUtil()

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.requires_grad:
                p.data.normal_()

    def leaf_transform(self, h):
        hbar = self.inside_compose.leaf_transform(h)
        hbar = UnitNorm()(hbar)
        return hbar

    def inside_pass(self, chart):
        length = chart['info']['length']

        outputs = {}
        for level in range(1, length):
            outputs[level] = self.inside_func(chart, level)
        return outputs

    def inside_func(self, chart, level):
        info = chart['info']
        batch_size, length, size = info['batch_size'], info['length'], info['size']
        L = length - level
        N = level

        component_lookup = InsideUtil.get_components(length, level)

        device = info['device']
        l_index = torch.tensor(component_lookup['l_idx'], dtype=torch.long, device=device)
        r_index = torch.tensor(component_lookup['r_idx'], dtype=torch.long, device=device)

        lh = torch.index_select(chart['inside_h'][:, 0], index=l_index, dim=1)
        rh = torch.index_select(chart['inside_h'][:, 0], index=r_index, dim=1)

        h = self.inside_compose(lh, rh).view(batch_size, L, N, size)
        hbar = h.sum(dim=2)
        hbar = UnitNorm()(hbar)

        ls = torch.index_select(chart['inside_s'][:, 0], index=l_index, dim=1)
        rs = torch.index_select(chart['inside_s'][:, 0], index=r_index, dim=1)

        xs = self.outside_score(lh, rh)
        s = (xs + ls + rs).view(batch_size, L, N, 1)
        sbar = s.sum(dim=2)

        # Fill chart.
        offset = get_offset_lookup(length)[level]
        chart['inside_h'][:, 0, offset:offset+L] = hbar
        chart['inside_s'][:, 0, offset:offset+L] = sbar

        outputs = {}
        outputs['xs'] = xs
        outputs['s'] = s

        return outputs

    def outside_pass(self, chart):
        length = chart['info']['length']

        outputs = {}
        for level in range(0, length-1)[::-1]:
            outputs[level] = self.outside_func(chart, level)
        return outputs

    def outside_func(self, chart, level):
        info = chart['info']
        batch_size, length, size = info['batch_size'], info['length'], info['size']
        L = length - level
        N = length - level - 1

        component_lookup = OutsideUtil.get_components(length, level)

        device = info['device']
        p_index = torch.tensor(component_lookup['p_idx'], dtype=torch.long, device=device)
        s_index = torch.tensor(component_lookup['s_idx'], dtype=torch.long, device=device)

        ph = torch.index_select(chart['outside_h'][:, 0], index=p_index, dim=1)
        sh = torch.index_select(chart['outside_h'][:, 0], index=s_index, dim=1)

        h = self.outside_compose(ph, sh).view(batch_size, L, N, size)
        hbar = h.sum(dim=2)
        hbar = UnitNorm()(hbar)

        ps = torch.index_select(chart['outside_s'][:, 0], index=p_index, dim=1)
        ss = torch.index_select(chart['outside_s'][:, 0], index=s_index, dim=1)

        xs = self.outside_score(ph, sh)
        s = (xs + ps + ss).view(batch_size, L, N, 1)
        sbar = s.sum(dim=2)

        # Fill chart.
        offset = get_offset_lookup(length)[level]
        chart['outside_h'][:, 0, offset:offset+L] = hbar
        chart['outside_s'][:, 0, offset:offset+L] = sbar

        outputs = {}
        outputs['xs'] = xs
        outputs['s'] = s

        return outputs

    def forward(self, h, run_args={}):
        device = h.device

        # Transform leaf.
        if run_args.get('leaf_transform', True):
            h = self.leaf_transform(h)
        batch_size, length, size = h.shape
        assert size == self.size

        # Build chart.
        init = {}
        init[('inside_h', 0, 0, length)] = h
        init[('outside_h', 0, -1, None)] = self.outside_h_root.view(1, 1, self.size).expand(batch_size, 1, self.size)
        chart = build_chart(batch_size, length, size, init=init, device=device)

        # Run inside / outside.
        outputs = {}
        outputs['chart'] = chart
        outputs['inside'] = self.inside_pass(chart)
        if run_args.get('outside', True):
            outputs['outside'] = self.outside_pass(chart)

        return outputs
