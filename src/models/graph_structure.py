"""Learned question-to-ability assignment with persistent pruning/freeze."""
import torch
from torch import nn
from .shared_graph import SharedGraphEncoder

class AbilityGraph(nn.Module):

    def __init__(self, n_concepts, abilities_per_concept, dim, pair_keys=(), temperature=1.0, relations=(), graph_layers=2, graph_initial_residual=0.1):
        super().__init__()
        if temperature <= 0 or abilities_per_concept < 1:
            raise ValueError('temperature and abilities_per_concept must be positive')
        self.n_concepts = n_concepts
        self.abilities_per_concept = abilities_per_concept
        self.n_abilities = n_concepts * abilities_per_concept
        self.temperature = temperature
        self.pair_keys = tuple(((int(q), int(c)) for (q, c) in pair_keys))
        if len(set(self.pair_keys)) != len(self.pair_keys):
            raise ValueError('Duplicate question-concept pairs')
        self.pair_to_row = {pair: row for (row, pair) in enumerate(self.pair_keys)}
        self.question_projection = nn.Linear(dim, dim, bias=False)
        self.ability_projection = nn.Linear(dim, dim, bias=False)
        nn.init.xavier_uniform_(self.question_projection.weight, gain=0.1)
        nn.init.xavier_uniform_(self.ability_projection.weight, gain=0.1)
        self.score_scale = dim ** (-0.5)
        shape = (len(self.pair_keys), self.abilities_per_concept)
        self.register_buffer('active_mask', torch.ones(shape, dtype=torch.bool))
        self.register_buffer('frozen_weights', torch.zeros(shape))
        self.register_buffer('frozen', torch.tensor(False))
        self.register_buffer('pair_q', torch.tensor([q for (q, _) in self.pair_keys], dtype=torch.long))
        self.register_buffer('pair_c', torch.tensor([c for (_, c) in self.pair_keys], dtype=torch.long))
        self.encoder = SharedGraphEncoder(dim, n_concepts, self.abilities_per_concept, relations, graph_layers, graph_initial_residual)

    def ability_ids(self, concepts):
        slots = torch.arange(self.abilities_per_concept, device=concepts.device)
        return concepts.unsqueeze(-1) * self.abilities_per_concept + slots

    def _all_live_weights(self, reader):
        if not self.pair_keys:
            return self.frozen_weights
        abilities = self.ability_ids(self.pair_c)
        q = self.question_projection(reader.question_embedding(self.pair_q))
        ability_vectors = self.ability_projection(reader.ability_embedding(abilities))
        scores = (q[:, None, :] * ability_vectors).sum(-1) * self.score_scale / self.temperature
        return torch.softmax(scores.masked_fill(~self.active_mask, -torch.inf), dim=-1)

    def encode(self, reader):
        weights = self.frozen_weights if bool(self.frozen) else self._all_live_weights(reader)
        nodes = self.encoder(reader, self.pair_q, self.pair_c, weights, self.ability_ids(self.pair_c))
        nodes['assignment_weights'] = weights
        return nodes

    @torch.no_grad()
    def prune_below(self, threshold, reader):
        if bool(self.frozen):
            raise RuntimeError('Cannot prune a frozen ability graph')
        if not 0 <= threshold < 1:
            raise ValueError('prune threshold must be in [0, 1)')
        weights = self._all_live_weights(reader)
        old_count = int(self.active_mask.sum().item())
        keep = self.active_mask & (weights >= threshold)
        if len(self.pair_keys):
            strongest = weights.masked_fill(~self.active_mask, -torch.inf).argmax(dim=1, keepdim=True)
            keep.scatter_(1, strongest, True)
        self.active_mask.copy_(keep)
        removed = old_count - int(self.active_mask.sum().item())
        return {'removed': removed, 'remaining': int(self.active_mask.sum().item()), 'pairs': len(self.pair_keys), 'single_candidate_pairs': int((self.active_mask.sum(dim=1) == 1).sum().item())}

    @torch.no_grad()
    def freeze(self, reader):
        if not bool(self.frozen):
            self.frozen_weights.copy_(self._all_live_weights(reader))
            self.frozen.fill_(True)
            self.question_projection.requires_grad_(False)
            self.ability_projection.requires_grad_(False)
        return {'pairs': len(self.pair_keys), 'remaining': int(self.active_mask.sum().item())}

    def restore_frozen_parameters(self):
        if bool(self.frozen) and True:
            self.question_projection.requires_grad_(False)
            self.ability_projection.requires_grad_(False)
