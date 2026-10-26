"""Shared model construction and metric helper."""
import torch
from .model import CoarseToFineKT

def resolve_device(config):
    requested = config.get('device', 'cpu')
    if requested not in {'cpu', 'cuda'}:
        raise ValueError(f'Unsupported device: {requested}')
    if requested == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('config requests CUDA, but torch.cuda.is_available() is False')
    return torch.device(requested)

def make_model(metadata, config, relations=None, pair_keys=()):
    settings = config['model_config']
    return CoarseToFineKT(metadata['n_questions'], metadata['n_concepts'], settings['hidden_dim'], settings['abilities_per_concept'], relations=relations, pair_keys=pair_keys, assignment_temperature=settings['assignment_temperature'], dropout=settings['dropout'], graph_layers=settings['graph_layers'], graph_initial_residual=settings['graph_initial_residual'])

def auc_score(labels, probabilities):
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ranked = sorted(enumerate(probabilities), key=lambda x: x[1])
    rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][1] == ranked[index][1]:
            end += 1
        avg_rank = (index + 1 + end) / 2
        rank_sum += sum((labels[ranked[j][0]] * avg_rank for j in range(index, end)))
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
