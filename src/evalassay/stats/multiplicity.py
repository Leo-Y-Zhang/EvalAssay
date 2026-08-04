"""Family-wise error control across the quantities an audit reports.

An audit tests several artifacts at once. Testing four hypotheses at the five
percent level and reporting whichever came out significant gives roughly a one
in five chance of inventing an artifact from noise. Since the entire point of
this tool is to stop people over-reading benchmark numbers, it would be absurd
for the tool itself to over-read its own.

Holm's step-down procedure is used rather than plain Bonferroni: it controls the
same family-wise error rate under arbitrary dependence between the tests, which
is essential here because the interventions are deliberately correlated, but is
uniformly more powerful.
"""

from __future__ import annotations

from collections.abc import Sequence


def holm_bonferroni(p_values: Sequence[float]) -> tuple[float, ...]:
    """Adjust p-values with Holm's step-down procedure.

    Args:
        p_values: Unadjusted p-values, in any order.

    Returns:
        Adjusted p-values in the same order as the input. Comparing each against
        the family-wise alpha controls the family-wise error rate.

    Raises:
        ValueError: If any p-value lies outside ``[0, 1]``.
    """
    for p in p_values:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p-value {p} outside [0, 1]")

    m = len(p_values)
    if m == 0:
        return ()

    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        scaled = (m - rank) * p_values[index]
        running = max(running, scaled)
        adjusted[index] = min(1.0, running)

    return tuple(adjusted)
