"""Map detached scale weights to existing supervision groups without changing logits."""


def supervision_group_weights(scale_weights, groups, supervision, output_count):
    """Use the mean member weight; symmetric groups preserve total loss weight.

    For four heads whose weights sum to four, all 15 mutation groups sum
    to 15, four singleton groups sum to four, and paper's five groups sum
    to five. Empty powerset entries get zero and produce no loss.
    """
    if scale_weights is None:
        return None
    if supervision not in ("mutation", "deep_supervision", "paper"):
        raise ValueError("uncertainty weights require mutation, deep_supervision or paper")
    if output_count != 4 or len(scale_weights) != output_count:
        raise ValueError("uncertainty weighting expects weights for four outputs")
    return [
        sum(scale_weights[index] for index in group) / len(group) if group else 0.0
        for group in groups
    ]
