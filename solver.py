"""Feasibility and fastest-lap bound solver.

Model
-----
For one rider we have laps 1..N.  For every lap the judge may have pinned one
reading (already reviewed) or every reading of that lap is still selectable.
A complete choice is feasible iff, with ``c_0 = 0`` and ``c_i`` the cumulative
milliseconds after lap ``i``::

    c_i - c_{i-1} >= min_lap_ms        for i = 1..N

The fastest lap of a complete choice is ``min_i(c_i - c_{i-1})``.  We need its
minimum and maximum over *all* complete feasible choices.

The implementation never enumerates the (exponential) cartesian product.  It
uses reachability between consecutive laps:

* forward reachability tells which readings can end a feasible prefix;
* backward reachability tells which readings can start a feasible suffix;
* the minimum possible fastest lap is the lightest edge that belongs to at
  least one complete feasible path (see notes below);
* the maximum possible fastest lap is the largest threshold for which a path
  using only edges >= threshold still exists; a monotone predicate, found by
  binary search.

Notes on the lower bound: for a complete path P, min-edge(P) is itself an edge
of P.  Therefore the minimum over paths of min-edge(P) equals the minimum
weight over all edges that occur on at least one complete path: if e is the
lightest such edge, the path through e has no edge lighter than e (by
minimality), hence its minimum edge is exactly weight(e).  This avoids
enumerating paths.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Iterable, Sequence

BREAKS = "BREAKS"
NOT_BREAK = "NOT_BREAK"
REVIEW = "REVIEW"
INFEASIBLE = "INFEASIBLE"

STATUS_LABELS = {
    BREAKS: "确定破纪录",
    NOT_BREAK: "确定未破纪录",
    REVIEW: "待核",
    INFEASIBLE: "无可行序列",
}


@dataclass(frozen=True)
class SolveResult:
    feasible: bool
    n_laps: int
    min_lap_ms: int
    record_ms: int
    lower_ms: int | None
    upper_ms: int | None
    status: str
    reason: str | None = None


def _normalise_candidates(
    lap_candidates: Iterable[Iterable[int]],
) -> list[tuple[int, ...]]:
    normalised: list[tuple[int, ...]] = []
    for index, values in enumerate(lap_candidates, start=1):
        unique = sorted({int(value) for value in values})
        if not unique:
            raise ValueError(f"第 {index} 圈没有候选读数")
        normalised.append(tuple(unique))
    if not normalised:
        raise ValueError("至少需要一圈读数")
    return normalised


def _forward_reachability(
    candidates: Sequence[Sequence[int]], threshold: int
) -> list[list[bool]]:
    """Return reachability from lap 0 (value 0) using edges >= threshold."""

    result: list[list[bool]] = []
    # Virtual lap 0.
    previous_values: Sequence[int] = (0,)
    previous_prefix = [1]  # prefix count of reachable previous values

    for values in candidates:
        reachable: list[bool] = []
        for value in values:
            cutoff = bisect_right(previous_values, value - threshold)
            reachable.append(cutoff > 0 and previous_prefix[cutoff - 1] > 0)
        result.append(reachable)

        prefix = [0] * len(values)
        running = 0
        for i, flag in enumerate(reachable):
            running += 1 if flag else 0
            prefix[i] = running
        previous_values = values
        previous_prefix = prefix

    return result


def _is_feasible(candidates: Sequence[Sequence[int]], threshold: int) -> bool:
    reachable = _forward_reachability(candidates, threshold)
    return any(reachable[-1])


def _backward_reachability(
    candidates: Sequence[Sequence[int]], threshold: int
) -> list[list[bool]]:
    """For every reading, whether a feasible suffix can start there."""

    n = len(candidates)
    backward: list[list[bool]] = [[] for _ in range(n)]
    backward[-1] = [True] * len(candidates[-1])

    for lap in range(n - 2, -1, -1):
        next_values = candidates[lap + 1]
        next_backward = backward[lap + 1]

        # suffix[i] = number of True in next_backward[i:]
        suffix = [0] * (len(next_values) + 1)
        for i in range(len(next_values) - 1, -1, -1):
            suffix[i] = suffix[i + 1] + (1 if next_backward[i] else 0)

        current: list[bool] = []
        for value in candidates[lap]:
            first = bisect_left(next_values, value + threshold)
            current.append(first < len(next_values) and suffix[first] > 0)
        backward[lap] = current

    return backward


def _first_failure_reason(
    candidates: Sequence[Sequence[int]], min_lap_ms: int
) -> str:
    reachability = _forward_reachability(candidates, min_lap_ms)

    failure = None
    for lap, flags in enumerate(reachability):
        if not any(flags):
            failure = lap
            break

    if failure is None:  # Defensive: callers only ask when infeasible.
        return "没有完整的可行读数序列。"

    if failure == 0:
        largest = max(candidates[0])
        return (
            f"无可行序列：第 1 圈最大读数为 {largest} ms，仍小于最短可信圈时 "
            f"{min_lap_ms} ms（首圈从 0 起算）。"
        )

    previous_reachable = [
        value
        for value, flag in zip(candidates[failure - 1], reachability[failure - 1])
        if flag
    ]
    smallest_previous = min(previous_reachable)
    largest_current = max(candidates[failure])
    difference = largest_current - smallest_previous
    return (
        f"无可行序列：第 {failure + 1} 圈无法衔接第 {failure} 圈的可达读数。"
        f"第 {failure} 圈最小可达累计为 {smallest_previous} ms，"
        f"第 {failure + 1} 圈最大候选累计为 {largest_current} ms，"
        f"差值 {difference} ms 小于最短可信圈时 {min_lap_ms} ms。"
    )


def _minimum_fastest_lap(
    candidates: Sequence[Sequence[int]], min_lap_ms: int
) -> int:
    """Minimum edge weight among edges lying on a complete feasible path."""

    forward = _forward_reachability(candidates, min_lap_ms)
    backward = _backward_reachability(candidates, min_lap_ms)

    best: int | None = None
    for lap, values in enumerate(candidates):
        if lap == 0:
            previous_values: Sequence[int] = (0,)
            previous_flags: Sequence[bool] = (True,)
        else:
            previous_values = candidates[lap - 1]
            previous_flags = forward[lap - 1]

        reachable_previous = [
            value for value, flag in zip(previous_values, previous_flags) if flag
        ]
        if not reachable_previous:
            continue

        for index, value in enumerate(values):
            if not backward[lap][index]:
                continue
            # Largest previous value still respecting the minimum lap time
            # gives the smallest edge weight for this reading.
            cutoff = bisect_right(reachable_previous, value - min_lap_ms)
            if cutoff == 0:
                continue
            edge = value - reachable_previous[cutoff - 1]
            if best is None or edge < best:
                best = edge

    if best is None:  # Only possible if the caller passed an infeasible set.
        raise ValueError("不可行序列没有最快圈上下限")
    return best


def _maximum_fastest_lap(
    candidates: Sequence[Sequence[int]], min_lap_ms: int
) -> int:
    """Largest threshold admitting a path where every edge is >= threshold."""

    low = min_lap_ms
    high = max(candidates[-1])
    if low > high:
        raise ValueError("不可行序列没有最快圈上下限")

    while low < high:
        middle = (low + high + 1) // 2
        if _is_feasible(candidates, middle):
            low = middle
        else:
            high = middle - 1
    return low


def solve(
    lap_candidates: Iterable[Iterable[int]],
    min_lap_ms: int,
    record_ms: int,
) -> SolveResult:
    """Compute feasibility, fastest-lap bounds and the record verdict.

    ``lap_candidates`` contains one iterable of selectable cumulative
    timestamps per lap.  A reviewed lap is represented by a single value.
    """

    if min_lap_ms <= 0:
        raise ValueError("最短可信圈时必须是正整数毫秒")
    if record_ms < 0:
        raise ValueError("现有纪录不能为负数")

    candidates = _normalise_candidates(lap_candidates)
    n = len(candidates)

    if not _is_feasible(candidates, min_lap_ms):
        return SolveResult(
            feasible=False,
            n_laps=n,
            min_lap_ms=min_lap_ms,
            record_ms=record_ms,
            lower_ms=None,
            upper_ms=None,
            status=INFEASIBLE,
            reason=_first_failure_reason(candidates, min_lap_ms),
        )

    lower = _minimum_fastest_lap(candidates, min_lap_ms)
    upper = _maximum_fastest_lap(candidates, min_lap_ms)

    # A record is broken only by a strictly faster lap.  Equality with the
    # existing record is "not broken".
    if upper < record_ms:
        status = BREAKS
    elif lower >= record_ms:
        status = NOT_BREAK
    else:
        status = REVIEW

    return SolveResult(
        feasible=True,
        n_laps=n,
        min_lap_ms=min_lap_ms,
        record_ms=record_ms,
        lower_ms=lower,
        upper_ms=upper,
        status=status,
    )
