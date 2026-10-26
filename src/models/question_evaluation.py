"""Differentiable all-in-one prediction shared by training and evaluation."""
import torch
import torch.nn.functional as F


def _group_logsumexp(values, group, n_groups):
    # Detached offsets only stabilize the exponential; all values keep gradients.
    indices = group[:, None].expand_as(values)
    maximum = values.new_full((n_groups, values.shape[1]), -torch.inf)
    maximum.scatter_reduce_(0, indices, values.detach(), reduce="amax", include_self=True)
    sums = values.new_zeros(maximum.shape).index_add(0, group,
                                                     (values - maximum[group]).exp())
    return maximum + sums.log()


def question_forward(model, sequences, warmup_k):
    # The expanded trajectory is the history. Candidate branches below never
    # advance that trajectory or consume another candidate's answer.
    history, nodes, index = model.encode_sequences(sequences)
    owners, positions, starts, groups, labels = ([] for _ in range(5))
    for owner, sequence in enumerate(sequences):
        items = sequence["interactions"]
        start = 0
        while start < len(items):
            first = items[start]
            order = first.get("order_id", start)
            end = start + 1
            while end < len(items) and items[end].get("order_id", end) == order:
                if (items[end]["question_raw"], items[end]["response"]) != (
                        first["question_raw"], first["response"]):
                    raise ValueError(f"Inconsistent question/answer for student {sequence.get('student_raw')}, order {order}")
                end += 1
            if start >= max(1, warmup_k - 1):
                group = len(labels)
                labels.append(first["response"])
                # Duplicate annotations do not give a KC extra fusion weight.
                seen_candidates = set()
                for position in range(start, end):
                    item = items[position]
                    key = tuple(item["concepts_raw"])
                    if key in seen_candidates:
                        continue
                    seen_candidates.add(key)
                    owners.append(owner)
                    positions.append(position)
                    starts.append(start)
                    groups.append(group)
            start = end
    device = index["questions"].device
    if not labels:
        empty = (sum(value.sum() for value in history.values()) * 0).expand(0)
        logits = {name: empty for name in ("question",)}
        return index["responses"].new_empty(0), {name: value.sigmoid() for name, value in logits.items()}, {
            "logits": logits}
    owner = torch.tensor(owners, device=device)
    position = torch.tensor(positions, device=device)
    start = torch.tensor(starts, device=device)
    # Pack candidate queries by student so attention shares one history tensor
    # per student, rather than replicating [time, dim] for every candidate.
    query_counts = [0] * len(sequences)
    slots = []
    for student in owners:
        slots.append(query_counts[student])
        query_counts[student] += 1
    slot = torch.tensor(slots, device=device)
    query_shape = (len(sequences), max(query_counts))
    cutoffs = torch.full(query_shape, -1, dtype=torch.long, device=device)
    cutoffs[owner, slot] = start - 1
    inputs, hidden = {}, {}
    for name, values in nodes.items():
        inputs[name] = values.new_zeros((*query_shape, values.shape[-1])).index_put(
            (owner, slot), values[owner, position])
        # Column s-1 has consumed rows 0..s-1, before the group's first answer.
        hidden[name] = values.new_zeros((*query_shape, values.shape[-1])).index_put(
            (owner, slot), history[name][owner, start - 1])
    inputs["questions"] = index["questions"].new_zeros(query_shape).index_put(
        (owner, slot), index["questions"][owner, position])
    inputs["history"], inputs["history_cutoffs"] = history, cutoffs
    _, candidates = model.reader.read_history(inputs, hidden)
    group = torch.tensor(groups, device=device)
    names = ("question",)
    values = torch.stack([candidates["logits"][name][owner, slot] for name in names], -1)
    # logit(mean(sigmoid(candidate_logits))), evaluated in log space so BCE
    # remains finite even when every candidate probability rounds to 0 or 1.
    fused = (_group_logsumexp(F.logsigmoid(values), group, len(labels)) -
             _group_logsumexp(F.logsigmoid(-values), group, len(labels)))
    logits = {name: fused[:, i] for i, name in enumerate(names)}
    cache = {"logits": logits}
    return index["responses"].new_tensor(labels), {
        name: value.sigmoid() for name, value in logits.items()}, cache


@torch.no_grad()
def question_predictions(model, sequences, warmup_k):
    labels, probabilities, _ = question_forward(model, sequences, warmup_k)
    return labels, probabilities["question"]
