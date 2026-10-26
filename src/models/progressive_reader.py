"""Independent causal history encoders with progressive query-time readout."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class HistoryAttention(nn.Module):
    """A shared-node query retrieves strictly available encoded history."""

    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)

    def forward(self, query, history, allowed):
        return torch.nn.functional.scaled_dot_product_attention(self.query(query).unsqueeze(1), self.key(history).unsqueeze(1), self.value(history).unsqueeze(1), attn_mask=allowed.unsqueeze(1), dropout_p=0.0).squeeze(1)

class ProgressiveReader(nn.Module):

    def __init__(self, dim, n_questions, n_concepts, n_abilities, dropout=0.0):
        super().__init__()
        self.question_embedding = nn.Embedding(n_questions, dim, padding_idx=0)
        self.concept_embedding = nn.Embedding(n_concepts, dim, padding_idx=0)
        self.ability_embedding = nn.Embedding(n_abilities, dim, padding_idx=0)
        self.correct_embedding = nn.Embedding(2, dim)
        self.difficulty = nn.Embedding(n_questions, 1, padding_idx=0)
        self.input_dropout = nn.Dropout(dropout)
        self.knowledge_gru = nn.GRU(3 * dim, dim, batch_first=True)
        self.ability_gru = nn.GRU(3 * dim, dim, batch_first=True)
        self.question_gru = nn.GRU(3 * dim, dim, batch_first=True)
        self.history_attention = nn.ModuleDict({name: HistoryAttention(dim) for name in ('knowledge', 'ability', 'question')})
        own_width = 3
        self.knowledge_map = nn.Sequential(nn.Linear(own_width * dim, dim), nn.LayerNorm(dim), nn.ReLU(), nn.Dropout(dropout))
        readout_width = 4
        self.ability_map = nn.Sequential(nn.Linear(readout_width * dim, dim), nn.LayerNorm(dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.question_map = nn.Sequential(nn.Linear(readout_width * dim, dim), nn.LayerNorm(dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.readout = nn.Linear(dim, 1, bias=False)
        nn.init.zeros_(self.difficulty.weight)

    def logit(self, z, questions):
        logits = self.readout(z).squeeze(-1)
        logits = logits - self.difficulty(questions).squeeze(-1)
        return logits

    @staticmethod
    def _run_gru(gru, values, lengths, hidden=None):
        packed = pack_padded_sequence(values, lengths, batch_first=True, enforce_sorted=False)
        (packed_output, _) = gru(packed, hidden)
        (output, _) = pad_packed_sequence(packed_output, batch_first=True, total_length=values.shape[1])
        return output

    def encode_history(self, inputs):
        """Each GRU uses its own node history, without another GRU's output."""
        previous = inputs['previous']
        initial = inputs.get('initial_hidden', {})
        previous_correct = self.correct_embedding(inputs['previous_response'].squeeze(-1).long())
        k_input = self.input_dropout(torch.cat((previous['knowledge'], previous_correct, previous['knowledge'] * previous_correct), -1))
        h_k = self._run_gru(self.knowledge_gru, k_input, inputs['lengths'], initial.get('knowledge'))
        a_input = self.input_dropout(torch.cat((previous['ability'], previous_correct, previous['ability'] * previous_correct), -1))
        h_a = self._run_gru(self.ability_gru, a_input, inputs['lengths'], initial.get('ability'))
        q_input = self.input_dropout(torch.cat((previous['question'], previous_correct, previous['question'] * previous_correct), -1))
        h_q = self._run_gru(self.question_gru, q_input, inputs['lengths'], initial.get('question'))
        return {'knowledge': h_k, 'ability': h_a, 'question': h_q}

    def read_history(self, inputs, hidden):
        """Retrieve causal evidence and refine the concept estimate for prediction."""
        memory = inputs.get("history", hidden)
        cutoffs = inputs.get("history_cutoffs")
        if cutoffs is None:
            state = hidden["knowledge"]
            cutoffs = torch.arange(state.shape[1], device=state.device)[None, :].expand(state.shape[0], -1)
        length = memory["knowledge"].shape[1]
        allowed = torch.arange(length, device=cutoffs.device)[None, None, :] <= cutoffs[..., None]
        retrieved = {name: attention(inputs[name], memory[name], allowed)
                     for name, attention in self.history_attention.items()}
        z = self.knowledge_map(torch.cat((hidden["knowledge"], inputs["knowledge"],
                                         retrieved["knowledge"]), -1))
        for name in ("ability", "question"):
            correction = getattr(self, name + "_map")(torch.cat(
                (hidden[name], inputs[name], z, retrieved[name]), -1))
            z = z + correction
        logits = self.logit(z, inputs["questions"])
        return {"question": logits.sigmoid()}, {"logits": {"question": logits}}

    def forward(self, inputs):
        return self.read_history(inputs, self.encode_history(inputs))
