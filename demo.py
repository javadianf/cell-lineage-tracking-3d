"""
Worked example: score three trackers of increasing quality on the same data.

Run:  python demo.py
"""

import random

from plantcell_metrics import evaluate, format_report


def make_truth(n_parents=200, division_rate=0.4, seed=0):
    """A synthetic transition with a realistic division fan-out."""
    rng = random.Random(seed)
    truth, next_child = {}, 1000

    for parent in range(1, n_parents + 1):
        if rng.random() < division_rate:
            # Mostly two daughters, a real minority of three and four.
            k = rng.choices([2, 3, 4], weights=[80, 15, 5])[0]
        else:
            k = 1
        truth[parent] = list(range(next_child, next_child + k))
        next_child += k

    return truth


def tracker_no_division_model(truth):
    """Commits every parent to one child. The classic failure."""
    return {p: c[:1] for p, c in truth.items()}


def tracker_pairs_only(truth):
    """Handles two-daughter divisions, collapses anything larger."""
    return {p: (c if len(c) <= 2 else c[:1]) for p, c in truth.items()}


def tracker_good(truth, error_rate=0.1, seed=1):
    """Mostly correct, drops one sibling from some larger groups."""
    rng = random.Random(seed)
    out = {}
    for p, c in truth.items():
        if len(c) > 2 and rng.random() < error_rate:
            out[p] = c[:-1]
        else:
            out[p] = list(c)
    return out


if __name__ == "__main__":
    truth = make_truth()
    n_div = sum(1 for c in truth.values() if len(c) > 1)
    print(f"synthetic transition: {len(truth)} parents, {n_div} dividing, "
          f"{sum(len(c) for c in truth.values())} children\n")

    for name, tracker in [
        ("no division model", tracker_no_division_model),
        ("pairs only", tracker_pairs_only),
        ("good tracker", tracker_good),
    ]:
        print(format_report(evaluate(tracker(truth), truth), title=name.upper()))
        print()
