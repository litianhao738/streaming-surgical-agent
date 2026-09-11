"""Overlap independent visual branches and merge Phase only after four-head repair."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from time import perf_counter

from surgical_agent.research.verification.phase_extension import apply_phase_choices
from surgical_agent.research.verification.prior_panel import TASKS, labels


def run_parallel_repair(h0, interaction, phase, image_count=3):
    """Callbacks collect evidence independently; neither receives the other's result.

    interaction receives a private copy of the original H0. Expected provider
    failures must be represented by callback results; programming errors escape.
    """
    labels(h0)
    begin = perf_counter()

    def timed(callback, *args):
        start = perf_counter() - begin
        value = callback(*args)
        end = perf_counter() - begin
        return value, start, end

    with ThreadPoolExecutor(max_workers=2) as workers:
        graph_future = workers.submit(timed, interaction, deepcopy(h0))
        phase_future = workers.submit(timed, phase)
        graph, graph_start, graph_end = graph_future.result()
        phase_result, phase_start, phase_end = phase_future.result()
    joined = perf_counter() - begin
    merge_start = perf_counter()
    final, decision = apply_phase_choices(graph["prediction"], phase_result["raw_reviews"], image_count)
    if any(final[t] != graph["prediction"][t] for t in TASKS):
        raise AssertionError("parallel Phase branch altered the graph repair")
    merge_seconds = perf_counter() - merge_start
    return {"graph": graph, "phase": phase_result, "final": final, "phase_decision": decision,
        "timing": {"graph_start": graph_start, "graph_end": graph_end, "graph_seconds": graph_end - graph_start,
            "phase_start": phase_start, "phase_end": phase_end, "phase_seconds": phase_end - phase_start,
            "join_seconds": joined, "merge_seconds": merge_seconds, "parallel_seconds": perf_counter() - begin}}
