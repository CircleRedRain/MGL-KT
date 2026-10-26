"""Shared graph and batched graph-to-sequence inputs for three GRUs."""
import numpy as np
import torch
from torch import nn
from .graph_structure import AbilityGraph
from .progressive_reader import ProgressiveReader

class CoarseToFineKT(nn.Module):

    def __init__(self, n_questions, n_concepts, dim, abilities_per_concept, relations=None, pair_keys=(), assignment_temperature=1.0, dropout=0.0, graph_layers=2, graph_initial_residual=0.1):
        super().__init__()
        self.dim = dim
        self.graph = AbilityGraph(n_concepts, abilities_per_concept, dim, pair_keys, assignment_temperature, relations or (), graph_layers, graph_initial_residual)
        self.reader = ProgressiveReader(dim, n_questions, n_concepts, self.graph.n_abilities, dropout)

    def _batch_indices(self, sequences):
        """Build graph addresses on CPU once per batch, using the pruning mask."""
        batch = len(sequences)
        length = max((len(s['interactions']) for s in sequences))
        concepts = max((len(item['concepts']) for s in sequences for item in s['interactions']))
        slots = self.graph.abilities_per_concept
        shape = (batch, length, concepts)
        ability_shape = shape + (slots,)
        questions = np.zeros((batch, length), dtype=np.int64)
        responses = np.zeros((batch, length), dtype=np.float32)
        valid = np.zeros((batch, length), dtype=np.bool_)
        concept_ids = np.zeros(shape, dtype=np.int64)
        concept_valid = np.zeros(shape, dtype=np.bool_)
        pair_rows = np.full(shape, -1, dtype=np.int64)
        active_slots = np.zeros(ability_shape, dtype=np.bool_)
        masks = self.graph.active_mask.detach().cpu().tolist()
        for (owner, sequence) in enumerate(sequences):
            for (position, item) in enumerate(sequence['interactions']):
                q = item['question']
                questions[owner, position] = q
                responses[owner, position] = item['response']
                valid[owner, position] = True
                for (branch, c) in enumerate(item['concepts']):
                    concept_ids[owner, position, branch] = c
                    concept_valid[owner, position, branch] = True
                    row = self.graph.pair_to_row.get((q, c), -1)
                    pair_rows[owner, position, branch] = row
                    enabled = (0,) if c == 0 else tuple(range(slots)) if row < 0 else tuple((slot for (slot, flag) in enumerate(masks[row]) if flag))
                    for slot in enabled:
                        active_slots[owner, position, branch, slot] = True
        device = self.reader.question_embedding.weight.device

        def transfer(array):
            return torch.from_numpy(array).to(device)
        return {'questions': transfer(questions), 'responses': transfer(responses), 'valid_mask': transfer(valid), 'concepts': transfer(concept_ids), 'concept_valid': transfer(concept_valid), 'pair_rows': transfer(pair_rows), 'active_slots': transfer(active_slots)}

    def _sequence_inputs(self, sequences):
        if not sequences or any((len(s['interactions']) < 2 for s in sequences)):
            raise ValueError('Prediction requires at least two interactions per student sequence')
        index = self._batch_indices(sequences)
        graph = self.graph
        nodes = graph.encode(self.reader)
        slots = graph.abilities_per_concept
        concept = index['concepts']
        branch = index['concept_valid']
        beta = branch.to(nodes['knowledge'].dtype) / branch.sum(-1, keepdim=True).clamp_min(1)
        row = index['pair_rows']
        if graph.pair_keys:
            weights = nodes['assignment_weights']
            a = weights[row.clamp_min(0)]
        else:
            a = nodes['ability'].new_zeros((*row.shape, slots))
        unknown = row < 0
        uniform = a.new_full((), 1.0 / slots)
        a = torch.where(unknown.unsqueeze(-1), uniform, a)
        zero_concept = (concept == 0) & branch
        one_hot = torch.nn.functional.one_hot(torch.zeros_like(concept), num_classes=slots).to(a.dtype)
        a = torch.where(zero_concept.unsqueeze(-1), one_hot, a)
        a = a * index['active_slots'].to(a.dtype)
        pi = beta.unsqueeze(-1) * a
        ability_ids = graph.ability_ids(concept)
        knowledge = (nodes['knowledge'][concept] * beta.unsqueeze(-1)).sum(-2)
        ability = (nodes['ability'][ability_ids] * pi.unsqueeze(-1)).sum((-3, -2))
        fallback_message = (nodes['ability_message'][ability_ids] * pi.unsqueeze(-1)).sum((-3, -2))
        fallback = torch.relu(nodes['question_base'][index['questions']] + fallback_message)
        known = ((row >= 0) | ~branch).all(-1)
        question = torch.where(known.unsqueeze(-1), nodes['question'][index['questions']], fallback)
        question = question * index['valid_mask'].unsqueeze(-1)
        nodes_by_step = {'knowledge': knowledge, 'ability': ability, 'question': question}
        current = {name: value[:, 1:] for (name, value) in nodes_by_step.items()}
        previous = {name: value[:, :-1] for (name, value) in nodes_by_step.items()}
        responses = index['responses']
        previous_response = responses[:, :-1].unsqueeze(-1)
        inputs = {**current, 'previous': previous, 'previous_response': previous_response, 'questions': index['questions'][:, 1:], 'lengths': [len(s['interactions']) - 1 for s in sequences]}
        return (inputs, nodes_by_step, index)

    def encode_sequences(self, sequences):
        """Encode observed history once, without reading every row as a query."""
        (inputs, nodes, index) = self._sequence_inputs(sequences)
        return (self.reader.encode_history(inputs), nodes, index)
