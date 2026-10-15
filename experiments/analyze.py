"""Final analysis + paper figures. Everything is generated; no manual editing.

  python -m experiments.analyze --raw results/raw/main --out results/analysis/main

Outputs (CSV + PNG + PDF): per_seed.csv, paired_differences.csv, paired_summary.csv,
primary_result.json, escape_summary.csv, final_eval.csv, curves.csv, triggers.csv,
learning_curves.*, aed_triggers.*, final_eval.*, analysis_summary.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from evaluation.diagnostics import bad_minimum_event  # noqa: E402
from evaluation.events import analyse_series  # noqa: E402
from evaluation.metrics import wilson  # noqa: E402
from evaluation.stats import (bootstrap_mean_ci, holm, paired_bootstrap_ci,  # noqa: E402
                              paired_permutation_pvalue)
from experiments import manifest as MF  # noqa: E402

NAN = float("nan")
PAIR_METRICS = ["final_winrate_uct400", "final_winrate_uct800", "last_periodic_winrate",
                "mean_periodic_winrate", "escaped"]


def net_win(row: dict) -> float:
    r = float(row["return_p0"])
    return 1.0 if (r > 0 if row["player0_agent"] == "net" else r < 0) else 0.0


def read_games(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as h:
        return list(csv.DictReader(h))


def _step(p: Path) -> int:
    return int(p.stem.rsplit("step", 1)[1])


def load_shard(sd: Path) -> dict:
    cfg = json.loads((sd / "config.json").read_text()) if (sd / "config.json").exists() else {}
    mp = sd / "metrics.json"
    metrics = json.loads(mp.read_text()) if mp.exists() else []
    periodic = []
    for p in sorted(sd.glob("eval_step*.csv"), key=_step):
        g = read_games(p)
        if g:
            periodic.append((_step(p), float(np.mean([net_win(r) for r in g])), len(g)))
    finals = {}
    for key in ("uct400", "uct800"):
        ps = sorted(sd.glob(f"final_{key}_step*.csv"), key=_step)
        if ps:
            g = read_games(ps[-1])
            wins = int(sum(net_win(r) for r in g))
            finals[key] = (wins / len(g), wins, len(g))
    return {"cfg": cfg, "metrics": metrics, "periodic": periodic, "finals": finals,
            "complete": (sd / "DONE").exists()}


def shard_events(sh: dict, ev: dict) -> dict:
    by_iter = {r["iter"]: r for r in sh["metrics"]}
    pts = [(s, w) for s, w, _ in sh["periodic"] if s in by_iter]
    out = {"weak_stable_iter": None, "escape_iter": None, "escaped": False, "bad_min_v2_iter": None}
    if not pts:
        return out
    iters = [s for s, _ in pts]
    wr = [w for _, w in pts]
    tv = [by_iter[s].get("tv") for s in iters]
    ents = [by_iter[s].get("entropy", 0.0) for s in iters]
    res = analyse_series(iters, tv, ents, wr, tv_t=ev["tv_t"], eps_h=ev["entropy_eps_h"],
                         wr_t=ev["wr_t"], consecutive=ev["consecutive"],
                         escape_pp=ev["escape_pp"], hold=ev["escape_hold"])
    out.update({k: res[k] for k in ("weak_stable_iter", "escape_iter", "escaped")})
    reps = [by_iter[s].get("repeat", 0.0) for s in iters]
    for k in range(len(iters)):                       # frozen protocol_v2 rule, for comparison
        if bad_minimum_event(tv[:k + 1], ents[:k + 1], wr[:k + 1], reps[:k + 1]):
            out["bad_min_v2_iter"] = iters[k]
            break
    return out


def seed_record(arm: str, seed: int, sh: dict, ev: dict) -> dict:
    m, per = sh["metrics"], sh["periodic"]
    last = m[-1] if m else {}
    f4, f8 = sh["finals"].get("uct400"), sh["finals"].get("uct800")
    ev_out = shard_events(sh, ev)
    trig = [r["iter"] for r in m if r.get("shifted")]
    dis = [r["disagreement"] for r in m if r.get("disagreement") is not None]

    def wil(f):
        return wilson(f[1], f[2]) if f else (NAN, NAN)
    rec = {
        "arm": arm, "arch": sh["cfg"].get("arch", ""), "mode": sh["cfg"].get("mode", ""),
        "seed": seed, "complete": sh["complete"], "n_iters": len(m),
        "param_count": sh["cfg"].get("param_count", ""),
        "final_vloss": last.get("vloss", NAN), "final_tv": last.get("tv", NAN),
        "final_entropy": last.get("entropy", NAN), "final_repeat": last.get("repeat", NAN),
        "final_disagreement": dis[-1] if dis else NAN,
        "last_periodic_winrate": per[-1][1] if per else NAN,
        "mean_periodic_winrate": float(np.mean([w for _, w, _ in per])) if per else NAN,
        "final_winrate_uct400": f4[0] if f4 else NAN, "final_n_uct400": f4[2] if f4 else 0,
        "final_uct400_wilson_lo": wil(f4)[0], "final_uct400_wilson_hi": wil(f4)[1],
        "final_winrate_uct800": f8[0] if f8 else NAN, "final_n_uct800": f8[2] if f8 else 0,
        "final_uct800_wilson_lo": wil(f8)[0], "final_uct800_wilson_hi": wil(f8)[1],
        "weak_stable_iter": ev_out["weak_stable_iter"], "escape_iter": ev_out["escape_iter"],
        "escaped": int(ev_out["escaped"]), "bad_min_v2_iter": ev_out["bad_min_v2_iter"],
        "n_triggers": len(trig), "first_trigger_iter": trig[0] if trig else None,
    }
    return rec


def discover(raw: Path, order: list[str]) -> dict[str, dict[int, Path]]:
    arms: dict[str, dict[int, Path]] = {}
    for d in sorted(p for p in raw.iterdir() if p.is_dir() and p.name != "gates"):
        seeds = {int(s.name.split("_")[1]): s for s in d.glob("seed_*") if s.is_dir()}
        if seeds:
            arms[d.name] = seeds
    return {a: arms[a] for a in sorted(arms, key=lambda x: (order.index(x) if x in order else 99, x))}


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or (list(rows[0]) if rows else ["empty"])
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def pair_list(m: dict, present: set[str]) -> list[tuple[str, str, str]]:
    prim = m["analysis"]["primary"]
    pairs = [("primary", prim["treatment"], prim["control"])]
    pairs += [("secondary", prim["treatment"], c) for c in m["analysis"]["secondary_controls"]]
    pairs += [("gnn_aed_vs_gnn_baseline", "gnn_aed", "gnn_baseline"),
              ("representation_gnn_vs_cnn", "gnn_baseline", "cnn_baseline")]
    return [p for p in pairs if p[1] in present and p[2] in present]


def paired_tables(recs: list[dict], pairs, n_perm: int, n_boot: int, rng_seed: int):
    by = {(r["arm"], r["seed"]): r for r in recs}
    long_rows, summary = [], []
    for kind, tr, co in pairs:
        seeds = sorted({s for (a, s) in by if a == tr} & {s for (a, s) in by if a == co})
        for metric in PAIR_METRICS:
            diffs = []
            for s in seeds:
                t, c = by[(tr, s)][metric], by[(co, s)][metric]
                d = t - c if not (_nan(t) or _nan(c)) else NAN
                long_rows.append({"kind": kind, "treatment": tr, "control": co, "metric": metric,
                                  "seed": s, "treatment_value": t, "control_value": c, "diff": d})
                if not _nan(d):
                    diffs.append(d)
            lo, hi = paired_bootstrap_ci(diffs, n_boot, rng_seed)
            summary.append({"kind": kind, "treatment": tr, "control": co, "metric": metric,
                            "n_pairs": len(diffs), "mean_diff": float(np.mean(diffs)) if diffs else NAN,
                            "ci95_lo": lo, "ci95_hi": hi,
                            "p_perm": paired_permutation_pvalue(diffs, n_perm, rng_seed),
                            "p_holm": NAN})
    sec = [i for i, r in enumerate(summary) if r["kind"] == "secondary"
           and r["metric"] == "final_winrate_uct400"]
    for i, p in zip(sec, holm([summary[i]["p_perm"] for i in sec])):
        summary[i]["p_holm"] = p
    return long_rows, summary


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def save_fig(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=160)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def curves(shards, out: Path, colors) -> None:
    rows = []
    for (arm, seed), sh in shards.items():
        for r in sh["metrics"]:
            for k in ("vloss", "tv", "entropy", "repeat", "disagreement", "Bt"):
                if r.get(k) is not None:
                    rows.append({"arm": arm, "seed": seed, "iter": r["iter"], "metric": k,
                                 "value": r[k]})
        for s, w, _ in sh["periodic"]:
            rows.append({"arm": arm, "seed": seed, "iter": s, "metric": "winrate_uct", "value": w})
    write_csv(out / "curves.csv", rows, ["arm", "seed", "iter", "metric", "value"])
    panels = [("winrate_uct", "win rate vs UCT-400"), ("tv", "policy TV"), ("entropy", "root entropy"),
              ("repeat", "opening repeat rate"), ("disagreement", "net vs reference disagreement")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4))
    for ax, (metric, title) in zip(axes, panels):
        for arm in sorted({r["arm"] for r in rows}):
            per_seed: dict[int, dict[int, float]] = {}
            for r in rows:
                if r["arm"] == arm and r["metric"] == metric:
                    per_seed.setdefault(r["seed"], {})[r["iter"]] = r["value"]
            if not per_seed:
                continue
            for d in per_seed.values():
                xs = sorted(d)
                ax.plot(xs, [d[x] for x in xs], color=colors[arm], alpha=0.15, lw=0.8)
            xs = sorted({x for d in per_seed.values() for x in d})
            mat = np.array([[d.get(x, np.nan) for x in xs] for d in per_seed.values()])
            mean = np.nanmean(mat, axis=0)
            sem = np.nanstd(mat, axis=0) / np.sqrt(np.maximum(1, (~np.isnan(mat)).sum(0)))
            ax.plot(xs, mean, color=colors[arm], lw=2, label=arm)
            ax.fill_between(xs, mean - sem, mean + sem, color=colors[arm], alpha=0.2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("iteration")
    axes[0].legend(fontsize=7)
    save_fig(fig, out, "learning_curves")


def triggers(shards, out: Path, colors) -> None:
    rows = [{"arm": a, "seed": s, "iter": r["iter"], "Bt": r.get("Bt", 0.0),
             "shifted": int(bool(r.get("shifted")))}
            for (a, s), sh in shards.items() for r in sh["metrics"]]
    write_csv(out / "triggers.csv", rows, ["arm", "seed", "iter", "Bt", "shifted"])
    arms = sorted({a for (a, _), sh in shards.items() if sh["cfg"].get("mode") in ("detector", "random")})
    if not arms:
        return
    fig, axes = plt.subplots(len(arms), 2, figsize=(10, 2.8 * len(arms)), squeeze=False)
    for i, arm in enumerate(arms):
        seeds = sorted(s for (a, s) in shards if a == arm)
        bmax = shards[(arm, seeds[0])]["cfg"].get("bmax")
        for j, s in enumerate(seeds):
            ms = shards[(arm, s)]["metrics"]
            xs = [r["iter"] for r in ms]
            axes[i][0].plot(xs, [r.get("Bt", 0.0) for r in ms], color=colors[arm], alpha=0.5, lw=1)
            fired = [r["iter"] for r in ms if r.get("shifted")]
            axes[i][1].scatter(fired, [j] * len(fired), marker="|", s=200, color=colors[arm])
        if bmax is not None and shards[(arm, seeds[0])]["cfg"].get("mode") == "detector":
            axes[i][0].axhline(bmax, ls="--", color="k", lw=0.8)
        axes[i][0].set_title(f"{arm}: B_t per seed")
        axes[i][1].set_title(f"{arm}: trigger raster")
        axes[i][1].set_yticks(range(len(seeds)))
        axes[i][1].set_yticklabels([str(s) for s in seeds], fontsize=7)
        axes[i][1].set_xlabel("iteration")
    save_fig(fig, out, "aed_triggers")


def final_eval_outputs(recs, out: Path, colors, rng_seed: int, n_boot: int) -> None:
    rows, arms = [], sorted({r["arm"] for r in recs})
    fig, ax = plt.subplots(figsize=(1.6 * max(3, len(arms)) + 2, 3.6))
    for i, arm in enumerate(arms):
        for off, key in ((-0.18, "final_winrate_uct400"), (0.18, "final_winrate_uct800")):
            vals = [r[key] for r in recs if r["arm"] == arm and not _nan(r[key])]
            lo, hi = bootstrap_mean_ci(vals, n_boot, rng_seed)
            rows.append({"arm": arm, "metric": key, "n_seeds": len(vals),
                         "mean": float(np.mean(vals)) if vals else NAN, "ci95_lo": lo, "ci95_hi": hi})
            ax.scatter([i + off] * len(vals), vals, color=colors[arm], alpha=0.5, s=14)
            if vals:
                ax.plot([i + off - 0.1, i + off + 0.1], [np.mean(vals)] * 2, color="k", lw=2)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("final win rate (left: UCT-400, right: UCT-800)")
    write_csv(out / "final_eval.csv", rows, ["arm", "metric", "n_seeds", "mean", "ci95_lo", "ci95_hi"])
    save_fig(fig, out, "final_eval")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=MF.DEFAULT_PATH)
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--allow-unlocked", action="store_true")
    ap.add_argument("--n-perm", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    a = ap.parse_args(argv)
    m = MF.load(a.manifest)
    MF.verify(m, require_locked=not a.allow_unlocked, check_code=False)
    ana = m["analysis"]["primary"]
    n_perm = a.n_perm or ana["n_perm"]
    n_boot = a.n_boot or ana["n_boot"]
    rng_seed = ana["rng_seed"]
    a.out.mkdir(parents=True, exist_ok=True)
    MF.copy_into(a.out, a.manifest)

    arms = discover(a.raw, list(m["arms"]))
    shards = {(arm, s): load_shard(p) for arm, seeds in arms.items() for s, p in seeds.items()}
    incomplete = [k for k, sh in shards.items() if not sh["complete"]]
    if incomplete and not a.allow_incomplete:
        raise SystemExit(f"incomplete shards (use --allow-incomplete): {incomplete[:5]}")
    recs = [seed_record(arm, s, sh, m["events"]) for (arm, s), sh in shards.items()]
    recs.sort(key=lambda r: (list(arms).index(r["arm"]), r["seed"]))
    write_csv(a.out / "per_seed.csv", recs)

    palette = plt.get_cmap("tab10")
    colors = {arm: palette(i % 10) for i, arm in enumerate(arms)}
    pairs = pair_list(m, set(arms))
    long_rows, summary = paired_tables(recs, pairs, n_perm, n_boot, rng_seed)
    write_csv(a.out / "paired_differences.csv", long_rows,
              ["kind", "treatment", "control", "metric", "seed", "treatment_value",
               "control_value", "diff"])
    write_csv(a.out / "paired_summary.csv", summary,
              ["kind", "treatment", "control", "metric", "n_pairs", "mean_diff", "ci95_lo",
               "ci95_hi", "p_perm", "p_holm"])
    prim = next((r for r in summary if r["kind"] == "primary"
                 and r["metric"] == ana["metric"]), None)
    primary = {"preregistered": True, "metric": ana["metric"], "test": ana["test"],
               "alpha": ana["alpha"], "result": prim,
               "reject_null": bool(prim and prim["n_pairs"] >= 2 and prim["p_perm"] < ana["alpha"])}
    (a.out / "primary_result.json").write_text(json.dumps(primary, indent=2, default=str) + "\n")

    esc = []
    for arm in arms:
        rs = [r for r in recs if r["arm"] == arm]
        n = len(rs)
        for key, label in (("escaped", "escape"), ("weak_stable_iter", "weak_stable")):
            k = sum(1 for r in rs if (r[key] == 1 if key == "escaped" else r[key] is not None))
            lo, hi = wilson(k, n) if n else (NAN, NAN)
            esc.append({"arm": arm, "event": label, "k": k, "n": n,
                        "fraction": k / n if n else NAN, "wilson_lo": lo, "wilson_hi": hi})
    write_csv(a.out / "escape_summary.csv", esc, ["arm", "event", "k", "n", "fraction",
                                                   "wilson_lo", "wilson_hi"])
    curves(shards, a.out, colors)
    triggers(shards, a.out, colors)
    final_eval_outputs(recs, a.out, colors, rng_seed, n_boot)
    (a.out / "analysis_summary.json").write_text(json.dumps({
        "manifest_hash": m["lock"]["manifest_hash"], "arms": {k: sorted(v) for k, v in arms.items()},
        "n_shards": len(shards), "incomplete_shards": [list(k) for k in incomplete],
        "n_perm": n_perm, "n_boot": n_boot, "pairs": [list(p) for p in pairs]}, indent=2) + "\n")
    print(f"wrote analysis to {a.out} ({len(shards)} shards, {len(incomplete)} incomplete)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
