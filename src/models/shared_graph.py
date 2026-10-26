"""Directed KC relations and undirected K-A/Q-A membership connections."""
import torch
from torch import nn
RELATION_TYPES = {'Successor': 0, 'Associative': 1}

class SharedGraphEncoder(nn.Module):

    def __init__(self, dim, n_concepts, abilities_per_concept, edges, layers=2, initial_residual=0.1):
        super().__init__()
        if not isinstance(layers, int) or layers < 1:
            raise ValueError('graph_layers must be a positive integer')
        self.layers = layers
        if not 0 <= initial_residual <= 1:
            raise ValueError('graph_initial_residual must be in [0, 1]')
        self.initial_residual = initial_residual
        self.edges = tuple((dict(edge) for edge in edges))
        for edge in self.edges:
            if edge['relation_type'] not in RELATION_TYPES:
                raise ValueError(f"Unknown relation type: {edge['relation_type']}")
        self.register_buffer('source', torch.tensor([e['source'] for e in edges], dtype=torch.long))
        self.register_buffer('target', torch.tensor([e['target'] for e in edges], dtype=torch.long))
        self.register_buffer('relation_type', torch.tensor([RELATION_TYPES[e['relation_type']] for e in edges], dtype=torch.long))
        concept_ids = torch.arange(1, n_concepts).repeat_interleave(abilities_per_concept)
        slots = torch.arange(abilities_per_concept).repeat(n_concepts - 1)
        self.register_buffer('owner_concept', concept_ids)
        self.register_buffer('owner_ability', concept_ids * abilities_per_concept + slots)
        self.theta = nn.Parameter(torch.zeros(len(edges)))
        self.register_buffer('static_frozen', torch.tensor(False))
        self.knowledge_self = nn.Linear(dim, dim, bias=False)
        self.relation_weight = nn.Parameter(torch.empty(len(RELATION_TYPES), dim, dim))
        for weight in self.relation_weight:
            nn.init.xavier_uniform_(weight)
        self.ability_self = nn.Linear(dim, dim, bias=False)
        self.knowledge_to_ability = nn.Linear(dim, dim, bias=False)
        self.question_self = nn.Linear(dim, dim, bias=False)
        self.ability_to_question = nn.Linear(dim, dim, bias=False)

    def freeze_static(self):
        self.static_frozen.fill_(True)
        self.theta.requires_grad_(False)

    def restore_frozen_parameters(self):
        if bool(self.static_frozen):
            self.theta.requires_grad_(False)

    def forward(self, reader, pair_q, pair_c, assignment_weights, ability_ids=None):
        """Synchronous rounds let KC relations reach A then Q in three layers.

        Each round reads only its input snapshot. Edge weights are computed once
        and shared across layers and both membership directions; transforms are
        also shared across layers. No answers/student states enter this graph.
        """
        h_k = reader.concept_embedding.weight
        h_a = reader.ability_embedding.weight
        h_q = reader.question_embedding.weight
        (initial_k, initial_a, initial_q) = (h_k, h_a, h_q)
        residual = self.initial_residual
        strength = torch.sigmoid(self.theta)
        normalizer = h_k.new_zeros(h_k.shape[0]).index_add(0, self.target, strength)
        degree_k = h_k.new_zeros(h_k.shape[0]).index_add(0, self.owner_concept, h_k.new_ones(self.owner_concept.shape[0]))
        owner_degree_a = h_a.new_zeros(h_a.shape[0]).index_add(0, self.owner_ability, h_a.new_ones(self.owner_ability.shape[0]))
        if pair_q.numel():
            slots = assignment_weights.shape[1]
            if ability_ids is None:
                ability_ids = pair_c[:, None] * slots + torch.arange(slots, device=pair_c.device)
            q_ids = pair_q[:, None].expand_as(ability_ids).reshape(-1)
            a_ids = ability_ids.reshape(-1)
            branches_per_q = torch.bincount(pair_q, minlength=h_q.shape[0])
            edge_weight = (assignment_weights / branches_per_q[pair_q, None]).reshape(-1)
            degree_q = h_q.new_zeros(h_q.shape[0]).index_add(0, q_ids, edge_weight)
            degree_a = h_a.new_zeros(h_a.shape[0]).index_add(0, a_ids, edge_weight)
        for _ in range(self.layers):
            knowledge = self.knowledge_self(h_k)
            if self.source.numel():
                transformed = torch.bmm(h_k[self.source].unsqueeze(1), self.relation_weight[self.relation_type]).squeeze(1)
                incoming = torch.zeros_like(knowledge).index_add(0, self.target, transformed * strength[:, None])
                knowledge = knowledge + incoming / normalizer.clamp_min(1)[:, None]
            incoming_k = torch.zeros_like(knowledge).index_add(0, self.owner_concept, self.knowledge_to_ability(h_a[self.owner_ability]))
            knowledge = knowledge + incoming_k / degree_k.clamp_min(1)[:, None]
            owner_messages = torch.zeros_like(h_a).index_add(0, self.owner_ability, self.knowledge_to_ability(h_k[self.owner_concept]))
            ability = self.ability_self(h_a) + owner_messages / owner_degree_a.clamp_min(1)[:, None]
            question_base = (1 - residual) * self.question_self(h_q) + residual * initial_q
            question = question_base
            ability_message = (1 - residual) * self.ability_to_question(h_a)
            if pair_q.numel():
                incoming_q = torch.zeros_like(question).index_add(0, q_ids, ability_message[a_ids] * edge_weight[:, None])
                incoming_a = torch.zeros_like(ability).index_add(0, a_ids, self.ability_to_question(h_q)[q_ids] * edge_weight[:, None])
                question = question + incoming_q / degree_q.clamp_min(1)[:, None]
                ability = ability + incoming_a / degree_a.clamp_min(1)[:, None]
            h_k = torch.relu((1 - residual) * knowledge + residual * initial_k)
            h_a = torch.relu((1 - residual) * ability + residual * initial_a)
            h_q = torch.relu(question)
        return {'knowledge': h_k, 'ability': h_a, 'question': h_q, 'question_base': question_base, 'ability_message': ability_message}
