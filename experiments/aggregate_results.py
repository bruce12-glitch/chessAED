"""Aggregate raw results into one table. Reads only results/raw/<exp>/seed_*/."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.diagnostics import bad_minimum_event, escaped, repeat_rate
from evaluation.metrics import bootstrap_ci, net_winrate, read_eval_csv, wilson


def seed_series(seed_dir: Path) -> tuple[list[dict], list[float]]:
    """Per-eval-step winrates in ckpt-step order + metrics rows."""
    metrics = json.loads((seed_dir / "metrics.json").read_text(encoding="utf-8"))
    evals = sorted(seed_dir.glob("eval_step*.csv"),
                   key=lambda p: int(p.stem.replace("eval_step", "")))
    wrs = []
    for e in evals:
        rows = read_eval_csv(e)
        wrs.append(net_winrate(rows))
    return metrics, wrs


def seed_escape(seed_dir: Path) -> dict:
    metrics, wrs = seed_series(seed_dir)
    tvs, ents, reps = [], [], []
    openings_all: list = []
    mi = 0
    event_at: int | None = None
    event_wr = 0.0
    eval_iters: list[int] = []
    for ei, e in enumerate(sorted(seed_dir.glob("eval_step*.csv"),
                                 key=lambda p: int(p.stem.replace("eval_step", "")))):
        step = int(e.stem.replace("eval_step", ""))
        eval_iters.append(step)
        while mi < len(metrics) and metrics[mi]["iter"] <= step:
            m = metrics[mi]
            tvs.append(m.get("tv", 0.0))
            ents.append(m.get("entropy", 0.0))
            openings_all.extend(m.get("openings", []))
            mi += 1
        reps.append(repeat_rate(openings_all[-64:] or [None]))
        if event_at is None and bad_minimum_event(tvs, ents, wrs[:ei + 1], reps):
            event_at = step
            event_wr = wrs[ei]
    return {"event_iter": event_at,
            "escaped": escaped(event_wr, wrs[eval_iters.index(event_at) + 1:])
            if event_at is not None else False,
            "winrates": [round(w, 3) for w in wrs]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("results/raw"))
    ap.add_argument("--output", type=Path, default=Path("results/tables/summary.md"))
    a = ap.parse_args()
    rows = []
    for exp_dir in sorted(p for p in a.raw.iterdir() if p.is_dir()):
        for seed_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
            metrics_p = seed_dir / "metrics.json"
            evals = sorted(seed_dir.glob("eval_step*.csv"))
            if not metrics_p.exists() or not evals:
                continue
            metrics = json.loads(metrics_p.read_text(encoding="utf-8"))
            games = [g for e in evals for g in read_eval_csv(e)]
            wr = net_winrate(games)
            lo, hi = wilson(sum(1 for g in games
                                if (float(g["return_p0"]) > 0) == (g["player0_agent"] == "net")),
                            len(games))
            blo, bhi = bootstrap_ci([1.0 if (float(g["return_p0"]) > 0) == (g["player0_agent"] == "net")
                                     else 0.0 for g in games])
            esc = seed_escape(seed_dir)
            rows.append({"experiment": exp_dir.name, "seed": seed_dir.name,
                         "iters": len(metrics), "final_vloss": metrics[-1].get("vloss"),
                         "games": len(games), "winrate": round(wr, 3),
                         "wilson": [round(lo, 3), round(hi, 3)],
                         "bootstrap95": [round(blo, 3), round(bhi, 3)],
                         "event_iter": esc["event_iter"], "escaped": esc["escaped"]})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", encoding="utf-8") as h:
        h.write("| experiment | seed | iters | final_vloss | games | winrate | wilson | bootstrap95 | event_iter | escaped |\n")
        h.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            h.write(f"| {r['experiment']} | {r['seed']} | {r['iters']} | {r['final_vloss']} | "
                    f"{r['games']} | {r['winrate']} | {r['wilson']} | {r['bootstrap95']} | "
                    f"{r['event_iter']} | {r['escaped']} |\n")
    exps: dict[str, list] = {}
    for r in rows:
        exps.setdefault(r["experiment"], []).append(r)
    for exp, rs in sorted(exps.items()):
        ev = sum(1 for r in rs if r["event_iter"] is not None)
        es = sum(1 for r in rs if r["escaped"])
        print(f"{exp}: n={len(rs)} events={ev} escaped={es}")
    print(f"wrote {a.output} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
