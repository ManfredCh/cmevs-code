from __future__ import annotations

import hashlib
from typing import Any

from _common import cell_set, safe_div, universe_cells, valid_candidates


def deterministic_random_key(candidate_id: str, seed: int = 2026) -> str:
    return hashlib.sha256(f"{seed}:{candidate_id}".encode("utf-8")).hexdigest()


def select_views(
    candidates: list[dict[str, Any]],
    method: str,
    budget: int,
    lambda_conflict: float = 0.35,
    probe_weight: float = 0.15,
    seed: int = 2026,
) -> tuple[list[dict[str, Any]], set[str]]:
    valid = valid_candidates(candidates)
    universe = universe_cells(valid)
    universe_size = max(1, len(universe))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    covered: set[str] = set()

    if method == "random_seeded":
        order = sorted(valid, key=lambda row: deterministic_random_key(str(row["candidate_id"]), seed))
        for rank, candidate in enumerate(order[:budget]):
            cells = cell_set(candidate)
            new_cells = cells - covered
            covered.update(new_cells)
            selected.append(_selected_row(candidate, rank, safe_div(len(new_cells), universe_size), 0.0))
        return selected, covered

    for rank in range(budget):
        best_candidate = None
        best_tuple = None
        best_gain = 0.0
        for candidate in valid:
            cid = str(candidate["candidate_id"])
            if cid in selected_ids:
                continue
            cells = cell_set(candidate)
            marginal_gain = safe_div(len(cells - covered), universe_size)
            probe = float(candidate.get("single_view_probe_coverage", 0.0))
            conflict = float(candidate.get("conflict_prior", 0.0))

            if method == "greedy_coverage":
                score = marginal_gain + 0.02 * probe - 0.01 * conflict
            elif method == "single_view_probe":
                score = probe + 0.05 * marginal_gain - 0.01 * conflict
            elif method == "low_conflict":
                score = marginal_gain - lambda_conflict * conflict
            elif method == "cm_evs":
                score = marginal_gain + probe_weight * probe - lambda_conflict * conflict
            else:
                raise ValueError(f"Unknown selection method: {method}")

            score_tuple = (score, marginal_gain, -conflict, cid)
            if best_tuple is None or score_tuple > best_tuple:
                best_tuple = score_tuple
                best_candidate = candidate
                best_gain = marginal_gain

        if best_candidate is None:
            break
        cid = str(best_candidate["candidate_id"])
        selected_ids.add(cid)
        covered.update(cell_set(best_candidate))
        selected.append(_selected_row(best_candidate, rank, best_gain, float(best_tuple[0])))

    return selected, covered


def summarize_selection(
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    covered: set[str],
    method: str,
    budget: int,
    lambda_conflict: float = 0.35,
    source_label: str | None = None,
) -> dict[str, Any]:
    valid = valid_candidates(candidates)
    universe = universe_cells(valid)
    scene_id = str(valid[0].get("scene_id", "unknown")) if valid else "unknown"
    source = source_label or (str(valid[0].get("source", "unknown")) if valid else "unknown")
    mean_conflict = safe_div(sum(float(row.get("conflict_prior", 0.0)) for row in selected), len(selected))
    mean_probe = safe_div(sum(float(row.get("single_view_probe_coverage", 0.0)) for row in selected), len(selected))
    runtime = sum(float(row.get("runtime_s", 0.0)) for row in selected)
    coverage = safe_div(len(covered), max(1, len(universe)))
    return {
        "source": source,
        "scene_id": scene_id,
        "method": method,
        "budget": budget,
        "selected_views": len(selected),
        "coverage": round(coverage, 6),
        "coverage_per_view": round(safe_div(coverage, len(selected)), 6),
        "mean_conflict_prior": round(mean_conflict, 6),
        "mean_probe_coverage": round(mean_probe, 6),
        "estimated_runtime_s": round(runtime, 6),
        "lambda_conflict": lambda_conflict,
        "selected_ids": ";".join(str(row["candidate_id"]) for row in selected),
    }


def _selected_row(candidate: dict[str, Any], rank: int, marginal_gain: float, score: float) -> dict[str, Any]:
    row = dict(candidate)
    row["rank"] = rank
    row["marginal_gain"] = round(float(marginal_gain), 6)
    row["score"] = round(float(score), 6)
    return row

