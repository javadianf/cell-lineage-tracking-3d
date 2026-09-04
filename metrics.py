# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Javadian. All rights reserved.
"""
Evaluation metrics for cell lineage tracking.

Standalone: no dependencies beyond the standard library. Drop it anywhere.

A tracking result is a mapping from each parent cell in one timepoint to the
list of child cells it became in the next. A cell that did not divide has one
child. A cell that divided has several. This module scores such a mapping
against ground truth.

The metrics use different denominators, and conflating them produces
comparisons that look meaningful and are not. They are spelled out here rather
than left implicit:

    recall              of all true children, the fraction given the right parent
    precision           of all assigned children, the fraction correct
    one_to_one_precision
                        of cells called non-dividing, the fraction correct
    division_perfect    of dividing parents, the fraction where EVERY daughter
                        is exactly right. All-or-nothing: two of three siblings
                        correct scores zero. The strictest, and the primary one,
                        because a lineage tree with a missing branch is wrong
    any_overlap         of dividing parents, the fraction with at least one
                        correct daughter. Separates "found the event,
                        mis-assigned a child" from "missed the event"
    child_accuracy      of all daughters of correctly-identified dividing
                        parents, the fraction placed correctly. The metric to
                        use when comparing against work that reports
                        per-daughter division recall
    missed_as_one_to_one
                        dividing parents committed to a single child. The
                        dominant failure mode of matchers without a division
                        model

Why division_perfect rather than a per-daughter rate as the headline. A lineage
is a tree. If a parent divided into three and the tracker finds two of them, the
tree is wrong, and averaging that to 0.67 hides it. Both are reported so the
comparison against either convention is available.

Usage:
    from plantcell_metrics import evaluate, format_report

    truth      = {10: [100, 101], 11: [102]}
    prediction = {10: [100],      11: [102]}
    print(format_report(evaluate(prediction, truth)))
"""

__version__ = "0.1.0"

__all__ = ["evaluate", "format_report", "invert", "dividing_parents",
           "MetricError"]


class MetricError(ValueError):
    """Raised when an input is malformed in a way that would silently skew a score."""


# ============================================================================
# HELPERS
# ============================================================================

def invert(parent_to_children):
    """
    {parent: [children]} to {child: parent}.

    Raises if a child appears under two parents. That is not a scoring edge
    case to be smoothed over: a cell has one parent, so a duplicate means the
    tracker produced an inconsistent result, and silently keeping the last one
    would inflate the score.
    """
    child_to_parent = {}
    for parent, children in parent_to_children.items():
        for child in children:
            if child in child_to_parent and child_to_parent[child] != parent:
                raise MetricError(
                    f"child {child!r} is assigned to both parent "
                    f"{child_to_parent[child]!r} and {parent!r}"
                )
            child_to_parent[child] = parent
    return child_to_parent


def dividing_parents(parent_to_children, min_children=2):
    """Parents with at least min_children. The denominator for division metrics."""
    return {p: list(c) for p, c in parent_to_children.items()
            if len(c) >= min_children}


def _rate(numerator, denominator):
    """Guard every ratio, so an empty transition scores 0.0 rather than raising."""
    return numerator / denominator if denominator else 0.0


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(prediction, truth, max_plausible_children=None):
    """
    Score a prediction against ground truth.

    prediction, truth   {parent: [children]}
    max_plausible_children
                        if given, ground-truth parents with more children than
                        this are excluded from every division denominator.
                        Real segmentation data contains annotation artifacts,
                        a parent recorded with thirty children for instance,
                        which cap an all-or-nothing metric no matter how good
                        the tracker is. Excluding them separates the tracking
                        error from the data error. Off by default, since
                        turning it on silently would make results
                        incomparable.

    Returns a flat dict of counts and rates.
    """
    truth_children = invert(truth)
    predicted_children = invert(prediction)

    excluded = set()
    if max_plausible_children is not None:
        excluded = {p for p, c in truth.items()
                    if len(c) > max_plausible_children}
        if excluded:
            truth = {p: c for p, c in truth.items() if p not in excluded}
            truth_children = {c: p for c, p in truth_children.items()
                              if p not in excluded}

    truth_divisions = dividing_parents(truth)
    predicted_divisions = dividing_parents(prediction)
    predicted_single = {p: c[0] for p, c in prediction.items() if len(c) == 1}

    # ---- overall, counted per child ----------------------------------------
    # A child left unassigned lowers recall without lowering precision. That
    # separation is what lets a conservative tracker and an aggressive one be
    # compared honestly.
    correct = sum(1 for child, parent in predicted_children.items()
                  if truth_children.get(child) == parent)
    assigned = sum(1 for child in predicted_children if child in truth_children)
    incorrect = assigned - correct

    precision = _rate(correct, assigned)
    recall = _rate(correct, len(truth_children))
    f1 = _rate(2 * precision * recall, precision + recall)

    # ---- one-to-one ---------------------------------------------------------
    correct_single = sum(1 for parent, child in predicted_single.items()
                         if truth_children.get(child) == parent)

    # ---- divisions ----------------------------------------------------------
    perfect = partial = 0
    perfect_by_degree, total_by_degree = {}, {}

    for parent, true_children in truth_divisions.items():
        degree = len(true_children)
        total_by_degree[degree] = total_by_degree.get(degree, 0) + 1

        predicted_set = set(predicted_divisions.get(parent, ()))
        if not predicted_set:
            continue

        true_set = set(true_children)
        if predicted_set == true_set:
            perfect += 1
            perfect_by_degree[degree] = perfect_by_degree.get(degree, 0) + 1
        elif predicted_set & true_set:
            partial += 1

    missed_as_single = sum(1 for p in truth_divisions if p in predicted_single)

    child_correct = child_total = 0
    for parent, predicted_set in predicted_divisions.items():
        if parent not in truth_divisions:
            continue
        true_set = set(truth_divisions[parent])
        for child in predicted_set:
            child_total += 1
            if child in true_set:
                child_correct += 1

    n_divisions = len(truth_divisions)

    return {
        "correct": correct,
        "incorrect": incorrect,
        "total_truth": len(truth_children),
        "total_predicted": len(predicted_children),
        "precision": precision,
        "recall": recall,
        "f1": f1,

        "n_single_predicted": len(predicted_single),
        "n_single_truth": sum(1 for c in truth.values() if len(c) == 1),
        "correct_single": correct_single,
        "one_to_one_precision": _rate(correct_single, len(predicted_single)),

        "n_divisions_truth": n_divisions,
        "n_divisions_predicted": len(predicted_divisions),
        "division_perfect": perfect,
        "division_partial": partial,
        "division_perfect_rate": _rate(perfect, n_divisions),
        "any_overlap_rate": _rate(perfect + partial, n_divisions),
        "missed_as_one_to_one": missed_as_single,
        "child_correct": child_correct,
        "child_total": child_total,
        "child_accuracy": _rate(child_correct, child_total),

        "perfect_by_degree": perfect_by_degree,
        "total_by_degree": total_by_degree,
        "n_excluded": len(excluded),
    }


def format_report(metrics, title="EVALUATION", by_degree=True):
    """Render metrics as a readable block."""
    m = metrics
    lines = ["=" * 64, title, "=" * 64]

    if m["n_excluded"]:
        lines.append(f"  {m['n_excluded']} implausible parents excluded")

    lines += [
        f"  overall      {m['correct']}/{m['total_truth']} children correct",
        f"               recall {100 * m['recall']:.1f}%   "
        f"precision {100 * m['precision']:.1f}%   F1 {m['f1']:.3f}",
        f"  one-to-one   predicted {m['n_single_predicted']} "
        f"(truth {m['n_single_truth']}), "
        f"precision {100 * m['one_to_one_precision']:.1f}%",
        f"  divisions    predicted {m['n_divisions_predicted']} "
        f"(truth {m['n_divisions_truth']})",
        f"               perfect {m['division_perfect']}/"
        f"{m['n_divisions_truth']} "
        f"({100 * m['division_perfect_rate']:.1f}%)   "
        f"partial {m['division_partial']}   "
        f"any-overlap {100 * m['any_overlap_rate']:.1f}%",
        f"               missed as one-to-one: {m['missed_as_one_to_one']}",
    ]

    if m["child_total"]:
        lines.append(
            f"               child accuracy within divisions: "
            f"{m['child_correct']}/{m['child_total']} "
            f"({100 * m['child_accuracy']:.1f}%)"
        )

    if by_degree and m["total_by_degree"]:
        lines.append("")
        lines.append("  division-perfect by number of daughters:")
        for degree in sorted(m["total_by_degree"]):
            total = m["total_by_degree"][degree]
            good = m["perfect_by_degree"].get(degree, 0)
            lines.append(f"    {degree} daughters: {good}/{total} "
                         f"({100 * good / total:.1f}%)")

    lines.append("=" * 64)
    return "\n".join(lines)
