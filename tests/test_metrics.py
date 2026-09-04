# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Javadian. All rights reserved.
"""
Tests for plantcell_metrics.

Runs with pytest, or standalone with `python tests/test_metrics.py`. No
dependencies beyond the standard library.

The cases here are the ones where a tracking metric is easy to get wrong: an
all-or-nothing division score that quietly gives partial credit, a precision
that counts unassigned cells as errors, a duplicate assignment that inflates a
score, and a division metric whose denominator shifts when the tracker changes.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics import MetricError, evaluate, invert


# A small transition: five parents, two of which divide.
TRUTH = {
    1: [10],
    2: [11],
    3: [12, 13],        # divides into two
    4: [14, 15, 16],    # divides into three
    5: [17],
}


def test_perfect_prediction():
    m = evaluate(TRUTH, TRUTH)
    assert m["recall"] == 1.0
    assert m["precision"] == 1.0
    assert m["division_perfect_rate"] == 1.0
    assert m["one_to_one_precision"] == 1.0
    assert m["missed_as_one_to_one"] == 0


def test_empty_prediction_scores_zero_without_raising():
    m = evaluate({}, TRUTH)
    assert m["recall"] == 0.0
    assert m["precision"] == 0.0
    assert m["division_perfect_rate"] == 0.0


def test_division_perfect_is_all_or_nothing():
    """Two of three siblings correct must score zero, not 0.67."""
    prediction = dict(TRUTH)
    prediction[4] = [14, 15]        # one sibling dropped

    m = evaluate(prediction, TRUTH)
    assert m["division_perfect"] == 1          # only parent 3 is perfect
    assert m["division_partial"] == 1          # parent 4 is partial
    assert m["division_perfect_rate"] == 0.5

    # The lenient companion still gives it credit, which is the point of
    # reporting both.
    assert m["any_overlap_rate"] == 1.0
    assert m["child_accuracy"] == 1.0          # every assigned child is right


def test_collapsed_divisions_are_counted_as_missed():
    """The dominant failure mode: a divider committed to a single child."""
    prediction = {p: c[:1] for p, c in TRUTH.items()}

    m = evaluate(prediction, TRUTH)
    assert m["division_perfect"] == 0
    assert m["missed_as_one_to_one"] == 2      # both dividers collapsed
    assert m["n_divisions_predicted"] == 0
    # One-to-one precision stays perfect: every cell it called non-dividing
    # really does have that parent. The failure is visible only in the
    # division metrics, which is why they are separate.
    assert m["one_to_one_precision"] == 1.0


def test_unassigned_children_lower_recall_not_precision():
    prediction = {1: [10], 2: [11]}            # three parents omitted entirely

    m = evaluate(prediction, TRUTH)
    assert m["precision"] == 1.0               # nothing assigned is wrong
    assert m["recall"] < 1.0                   # but most children are missing


def test_wrong_parent_lowers_both():
    prediction = dict(TRUTH)
    prediction[1] = [11]                       # steals parent 2's child
    prediction[2] = [10]

    m = evaluate(prediction, TRUTH)
    assert m["precision"] < 1.0
    assert m["recall"] < 1.0
    assert m["incorrect"] == 2


def test_duplicate_child_assignment_raises():
    """A child under two parents is an inconsistent result, not a scoring case."""
    bad = {1: [10], 2: [10]}
    try:
        invert(bad)
    except MetricError:
        pass
    else:
        raise AssertionError("expected MetricError for a duplicated child")


def test_exclusion_changes_the_denominator():
    """Implausible parents cap an all-or-nothing metric; excluding them is opt-in."""
    truth = dict(TRUTH)
    truth[6] = list(range(100, 130))           # 30 children, an artifact

    included = evaluate(TRUTH, truth)
    excluded = evaluate(TRUTH, truth, max_plausible_children=4)

    assert included["n_divisions_truth"] == 3
    assert excluded["n_divisions_truth"] == 2
    assert excluded["n_excluded"] == 1
    # The tracker did not change, only the denominator did.
    assert excluded["division_perfect_rate"] > included["division_perfect_rate"]


def test_by_degree_breakdown_sums_to_total():
    prediction = dict(TRUTH)
    prediction[4] = [14, 15]

    m = evaluate(prediction, TRUTH)
    assert sum(m["total_by_degree"].values()) == m["n_divisions_truth"]
    assert sum(m["perfect_by_degree"].values()) == m["division_perfect"]


def test_extra_predicted_division_does_not_inflate():
    """Predicting a division that did not happen must not create credit."""
    prediction = dict(TRUTH)
    prediction[1] = [10, 17]                   # invents a division, steals 17
    del prediction[5]

    m = evaluate(prediction, TRUTH)
    assert m["division_perfect"] == 2          # unchanged, 3 and 4 still right
    assert m["n_divisions_predicted"] == 3     # but one more was claimed
    assert m["incorrect"] >= 1


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if not name.startswith("test_") or not callable(function):
            continue
        try:
            function()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}: {exc}")

    print(f"\n{'all tests passed' if not failures else f'{failures} failed'}")
    sys.exit(1 if failures else 0)
