"""Training protocol shared by the Lightning runner and focused regression checks."""
import torch

def validate_schedule(config):
    schedule = config['schedule']
    (total, start) = (config['max_epochs'], schedule['prune_start_epoch'])
    if not isinstance(total, int) or not isinstance(start, int) or (not 1 <= start < total):
        raise ValueError('prune_start_epoch must precede the final training epoch')
    if not isinstance(schedule['stable_epochs'], int) or schedule['stable_epochs'] < 1:
        raise ValueError('stable_epochs must be a positive integer')
    for name in ('weight_prune_threshold', 'weight_change_tolerance'):
        if not isinstance(schedule[name], (int, float)) or not 0 <= schedule[name] < 1:
            raise ValueError(f'{name} must be in [0, 1)')

@torch.no_grad()
def update_structure(model, schedule, monitor, epoch):
    if bool(model.graph.frozen) or epoch < schedule['prune_start_epoch']:
        return None
    pruning = model.graph.prune_below(schedule['weight_prune_threshold'], model.reader)
    current_mask = model.graph.active_mask.detach().cpu().clone()
    current_weights = model.graph._all_live_weights(model.reader).detach().cpu()
    previous_mask = monitor['previous_mask']
    mask_changed = previous_mask is not None and (not torch.equal(current_mask, previous_mask))
    changes = None if previous_mask is None else (current_weights - monitor['previous_weights']).abs()
    active_changes = None if changes is None else changes[current_mask]
    mean_change = None if active_changes is None else active_changes.mean().item()
    max_change = None if active_changes is None else active_changes.max().item()
    monitor['stable_epochs'] = monitor['stable_epochs'] + 1 if mean_change is not None and (not mask_changed) and (mean_change <= schedule['weight_change_tolerance']) else 0
    monitor['previous_mask'] = current_mask
    monitor['previous_weights'] = current_weights
    all_single = bool(current_mask.sum(dim=1).eq(1).all())
    reason = 'all_single' if all_single else 'stable_weights' if monitor['stable_epochs'] >= schedule['stable_epochs'] else None
    result = {'pruning': pruning, 'mask_changed': mask_changed, 'mean_weight_change': mean_change, 'max_weight_change': max_change, 'stable_epochs': monitor['stable_epochs'], 'all_single': all_single, 'freeze_reason': reason}
    if reason is not None:
        result['freeze'] = model.graph.freeze(model.reader)
        model.graph.encoder.freeze_static()
    return result
