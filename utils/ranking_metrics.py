"""Threshold-free ranking metrics.

Fidelity is always measured at some decision threshold, so it mixes two
different things: how well the model separates bright from dark, and how well
the threshold happens to be placed.  AUC depends only on the ranking, which
makes it the right tool for asking "are these weights any good" independently
of calibration.
"""

import numpy as np


def average_ranks(values):
    """Ranks with ties averaged.

    Softmax scores saturate hard at 0 and 1 for the long exposures, so the tie
    handling is not cosmetic here -- ordinary ranks would inflate the AUC.
    """
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)

    start = 0
    while start < values.size:
        stop = start
        while stop + 1 < values.size and sorted_values[stop + 1] == sorted_values[start]:
            stop += 1
        ranks[order[start : stop + 1]] = 0.5 * (start + stop) + 1.0
        start = stop + 1

    return ranks


def per_site_auc(scores, labels):
    """Mann-Whitney AUC per site; NaN where a site has only one class."""
    labels = np.asarray(labels)
    n_sites = labels.shape[1]
    aucs = np.full(n_sites, np.nan, dtype=np.float64)

    for site_index in range(n_sites):
        site_scores = np.asarray(scores[:, site_index], dtype=np.float64)
        bright = labels[:, site_index] > 0.5
        n_bright = int(bright.sum())
        n_dark = int((~bright).sum())
        if n_bright == 0 or n_dark == 0:
            continue
        ranks = average_ranks(site_scores)
        aucs[site_index] = (
            float(ranks[bright].sum()) - n_bright * (n_bright + 1) / 2.0
        ) / (n_bright * n_dark)

    return aucs


def mean_auc(scores, labels):
    aucs = per_site_auc(scores, labels)
    if not np.isfinite(aucs).any():
        return float("nan")
    return float(np.nanmean(aucs))
