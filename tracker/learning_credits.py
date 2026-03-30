"""
Proof-of-Learning (PoL) credit scoring — tracker-side only.

Credits reward alignment between (a) submitted shard-level signals and (b) global model
improvement after FedAvg. This is not proof against malicious clients; it is a practical
incentive layer assuming honest-but-calibrated metadata (shard_eval_acc, steps_completed).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class CreditBreakdown:
    """One credit grant with explainable components (for APIs / dashboard)."""

    credits: float
    phase: str
    detail: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"credits": self.credits, "phase": self.phase, **self.detail}


def parse_shard_index(task_id: str) -> int:
    if task_id.startswith("shard-"):
        try:
            return int(task_id.split("-", 1)[1])
        except ValueError:
            pass
    return 0


def shard_rarity_multiplier(task_id: str, num_shards: int = 5) -> float:
    """
    Deterministic shard weight — stands in for "harder / less redundant" slices when
    true class-rarity is unavailable. Middle indices get a modest bump.
    """
    i = parse_shard_index(task_id)
    n = max(2, num_shards)
    center = (n - 1) / 2.0
    dist = abs(i - center) / max(center, 1e-6)
    return 1.0 + 0.06 * (1.0 - dist)


def reputation_multiplier(reputation: float) -> float:
    """Maps 0..100 reputation → ~0.55..1.65 credit multiplier."""
    r = max(0.0, min(100.0, reputation))
    return 0.55 + (r / 100.0) * 1.1


def apply_streak_bonus(credits: float, positive_streak: int, *, threshold: int = 3, bonus: float = 1.05) -> float:
    if positive_streak >= threshold:
        return credits * bonus
    return credits


def anti_spam_scale(
    steps_completed: int,
    shard_eval_acc: Optional[float],
    train_acc_running: Optional[float],
    *,
    min_steps: int = 8,
) -> tuple[float, list[str]]:
    """Scale down suspiciously tiny or inconsistent submissions."""
    reasons: list[str] = []
    scale = 1.0
    if steps_completed < min_steps:
        scale *= 0.2
        reasons.append("few_steps")
    if shard_eval_acc is not None and train_acc_running is not None:
        if train_acc_running - shard_eval_acc > 28.0:
            scale *= 0.55
            reasons.append("train_much_above_shard_eval")
        if shard_eval_acc - train_acc_running > 40.0:
            scale *= 0.55
            reasons.append("shard_eval_much_above_train")
    return scale, reasons


def interim_submit_credit(
    *,
    baseline_val_acc: Optional[float],
    shard_eval_acc: Optional[float],
    steps_completed: int,
    task_id: str,
    reputation: float,
    positive_streak: int,
    train_acc_running: Optional[float] = None,
) -> CreditBreakdown:
    """
    Reward per checkpoint when the worker reports shard_eval_acc vs last *global* val accuracy
    (proxy for progress direction before the next FedAvg).
    """
    if baseline_val_acc is None or shard_eval_acc is None:
        return CreditBreakdown(0.0, "interim_skip", {"reason": "missing_baseline_or_shard_eval"})

    delta = float(shard_eval_acc) - float(baseline_val_acc)
    steps = max(1, int(steps_completed))
    efficiency = delta / float(steps)
    rarity = shard_rarity_multiplier(task_id)
    spam_scale, spam_reasons = anti_spam_scale(steps_completed, shard_eval_acc, train_acc_running)

    pos_term = max(0.0, delta) * rarity * 0.14
    eff_term = max(0.0, efficiency) * rarity * 9.0
    raw = 0.62 * pos_term + 0.38 * eff_term

    penalty = 0.0
    if delta < -0.35:
        penalty = min(5.0, abs(delta) * 0.18)
        raw -= penalty

    raw *= spam_scale
    raw *= reputation_multiplier(reputation)
    raw = apply_streak_bonus(raw, positive_streak)

    return CreditBreakdown(
        round(raw, 4),
        "interim_submit",
        {
            "delta_vs_global_val": round(delta, 4),
            "efficiency_per_step": round(efficiency, 6),
            "rarity": round(rarity, 4),
            "spam_scale": spam_scale,
            "spam_reasons": spam_reasons,
            "penalty": round(penalty, 4),
            "reputation_mult": round(reputation_multiplier(reputation), 4),
        },
    )


def round_pool_distribution(
    *,
    old_val_acc: Optional[float],
    new_val_acc: Optional[float],
    shard_rows: list[dict[str, Any]],
) -> dict[str, CreditBreakdown]:
    """
    After FedAvg, distribute (or penalize) based on true global val movement.

    shard_rows: [{"worker_id", "task_id", "eval_acc"}, ...]
    """
    out: dict[str, CreditBreakdown] = {}
    if not shard_rows:
        return out

    if old_val_acc is None or new_val_acc is None:
        for row in shard_rows:
            wid = row.get("worker_id")
            if wid:
                out[str(wid)] = CreditBreakdown(0.0, "round_skip", {"reason": "missing_val_metrics"})
        return out

    global_delta = float(new_val_acc) - float(old_val_acc)
    improve_pool = max(0.0, global_delta) * 3.0
    degrade_penalty = max(0.0, -global_delta) * 1.4

    weighted: list[tuple[str, float, str]] = []
    for row in shard_rows:
        wid = row.get("worker_id")
        if not wid:
            continue
        tid = str(row.get("task_id", "shard-0"))
        ev = row.get("eval_acc")
        rarity = shard_rarity_multiplier(tid)
        if ev is None:
            w = 0.35 * rarity
        else:
            local_gain = max(0.0, float(ev) - float(old_val_acc))
            w = (0.55 * local_gain + 0.35) * rarity
        weighted.append((str(wid), w, tid))

    total_w = sum(w for _, w, __ in weighted) or 1.0

    credits_by_worker: dict[str, float] = defaultdict(float)
    detail_by_worker: dict[str, dict[str, Any]] = {}

    if improve_pool > 0.0:
        for wid, w, tid in weighted:
            share = improve_pool * (w / total_w)
            rarity = shard_rarity_multiplier(tid)
            credits_by_worker[wid] += share
            prev = detail_by_worker.get(wid, {"shards": [], "global_delta": round(global_delta, 4)})
            prev["shards"].append({"task_id": tid, "weight": round(w, 4), "rarity": round(rarity, 4), "share": round(share, 4)})
            detail_by_worker[wid] = prev

    if degrade_penalty > 0.0 and global_delta < -0.08:
        unique_workers = list({wid for wid, _, __ in weighted})
        per = degrade_penalty / max(1, len(unique_workers))
        for wid in unique_workers:
            credits_by_worker[wid] = credits_by_worker.get(wid, 0.0) - per
            d = detail_by_worker.setdefault(wid, {"shards": [], "global_delta": round(global_delta, 4)})
            d["global_penalty"] = round(-per, 4)

    for wid, total in credits_by_worker.items():
        detail = detail_by_worker.get(wid, {})
        out[wid] = CreditBreakdown(round(total, 4), "round_global_gain", detail)

    # Workers in shard_rows who got nothing (edge case)
    for row in shard_rows:
        wid = row.get("worker_id")
        if wid and str(wid) not in out:
            out[str(wid)] = CreditBreakdown(0.0, "round_skip", {"reason": "no_weight_row"})

    return out


def update_reputation(prev: float, signed_credit: float) -> float:
    """EMA: nudge reputation from recent signed credit (clamped signal)."""
    prev = max(0.0, min(100.0, prev))
    signal = max(-6.0, min(6.0, signed_credit))
    blended = prev * 0.88 + 8.0 * signal * 0.12
    return max(0.0, min(100.0, blended))
