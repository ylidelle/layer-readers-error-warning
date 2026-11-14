"""SPRINT analysis (2026-11-14; rewritten after Alexander's and Lucien's reviews).

  analyze.py select   DEVELOPMENT set only (all 300 rows, made under the CURRENT inference stamp). For each reader,
                      every (layer 0..30, measure, direction) is scored; the highest development AUROC is kept.
                      Ties: lower layer, then measure name in the order p1, ent, margin; an AUROC of exactly 0.5
                      counts as direction +1. Writes out/picks.json with the stamp it was made under.
  analyze.py final    FINAL set, once. Requires the seal, the final rows stamped with it, and a raw output file
                      whose hash equals the one recorded when the run completed.

Orientation, used everywhere: AUROC = probability that a randomly chosen CORRECT answer gets a HIGHER score than a
randomly chosen WRONG one (ties half). Scores are oriented so that higher means 'more likely correct'.
Label: correct_em (exact match). correct_contain is a labelled sensitivity check only."""
import sys, json
import numpy as np
import config, seal
sys.stdout.reconfigure(encoding="utf-8")

def ranks(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x)); r[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1: r[m] = r[m].mean()
    return r

def auroc(score, y):
    y = np.asarray(y, bool); n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0: return float("nan")
    return float((ranks(score)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def said_scores(rows):
    return np.array([-1.0 if r["said"] is None else r["said"] for r in rows])   # invalid reply = below every valid value

def signal(rows, reader, layer, measure):
    return np.array([r["read"][reader][str(layer)][measure] for r in rows])

def verdict(lo, hi, margin=config.MARGIN):
    """Two separate statements from ONE interval (A minus B). Boundaries are strict: an end exactly on 0 or on the
    margin leaves that statement unresolved."""
    direction = "A higher" if lo > 0 else "B higher" if hi < 0 else "direction unresolved"
    if lo > -margin and hi < margin: size = "practically equivalent within the margin"
    elif lo > margin: size = "A better by more than the margin"
    elif hi < -margin: size = "B better by more than the margin"
    else: size = "practical size unresolved"
    return direction, size

def verified_dev():
    """Development rows, only if the run COMPLETED under the current inference stamp and the file has not changed
    since its completion hash was recorded. Returns (rows, stamp, output sha256)."""
    import os
    stamp = seal.inference_stamp()
    side_path = seal.out_path("dev").replace(".jsonl", ".manifest.json")
    if not os.path.exists(side_path): raise seal.Refused("no development run manifest")
    side = json.load(open(side_path, encoding="utf-8"))
    if side.get("stamp") != stamp: raise seal.Refused("the development run was made under a different inference stamp")
    if side.get("rows") != config.N_ROWS["dev"] or not side.get("output_sha"):
        raise seal.Refused("the development run is not recorded as complete")
    sha = seal.sha_file(seal.out_path("dev"))
    if sha != side["output_sha"]: raise seal.Refused("development output does not match the hash recorded when the run completed")
    return seal.load_complete("dev", stamp), stamp, sha

def select():
    rows, stamp, dev_sha = verified_dev()
    y = np.array([r["correct_em"] for r in rows]); yc = np.array([r["correct_contain"] for r in rows])
    print(f"development set: {len(rows)} questions | exact match: {int(y.sum())} correct, {int((~y).sum())} wrong "
          f"| containment label: {int(yc.sum())} correct | the two labels disagree on {int((y != yc).sum())}")
    sv = [r["said"] for r in rows]
    print(f"said: {sum(v is None for v in sv)} invalid replies; distinct valid values {sorted({v for v in sv if v is not None})[:15]}")
    print(f"baselines on dev: p_out {auroc([r['p_out'] for r in rows], y):.3f} | said {auroc(said_scores(rows), y):.3f}")
    print(f"first token agrees with a plain forward pass on {sum(r['first_id'] == r['fwd_top'] for r in rows)} of {len(rows)}")
    order = {m: i for i, m in enumerate(config.MEASURES)}
    picks = {}
    for rd in config.READERS:
        cands = []
        for L in config.LAYERS:
            for m in config.MEASURES:
                a = auroc(signal(rows, rd, L, m), y)
                cands.append((max(a, 1 - a), L, m, 1 if a >= 0.5 else -1))
        cands.sort(key=lambda c: (-c[0], c[1], order[c[2]]))
        b = cands[0]
        picks[rd] = {"layer": b[1], "measure": b[2], "direction": b[3], "dev_auroc": round(b[0], 4)}
        print(f"{rd:5s} pick: layer {b[1]:2d} {b[2]:6s} direction {b[3]:+d} dev AUROC {b[0]:.3f} | next: "
              + ", ".join(f"L{c[1]} {c[2]} {c[0]:.3f}" for c in cands[1:4])
              + f" | fixed L{config.FIXED_LAYER} {config.FIXED_MEASURE}: {auroc(signal(rows, rd, config.FIXED_LAYER, config.FIXED_MEASURE), y):.3f}")
    blob = json.dumps({"inference_stamp": stamp, "dev_output_sha": dev_sha, "label": "correct_em", "picks": picks}, sort_keys=True, indent=1).encode()
    open(config.BASE + "out/picks.json", "wb").write(blob)
    print("picks.json written under stamp", stamp[:16], "| sha256", seal.sha_file(config.BASE + "out/picks.json"))

def final():
    seal_sha = seal.require_sealed()
    side = json.load(open(seal.out_path("final").replace(".jsonl", ".manifest.json"), encoding="utf-8"))
    if side.get("stamp") != seal_sha: raise seal.Refused("the final run was not made under this seal")
    if side.get("output_sha") != seal.sha_file(seal.out_path("final")):
        raise seal.Refused("final output does not match the hash recorded when the run completed")
    rows = seal.load_complete("final", seal_sha)
    pk = json.load(open(config.BASE + "out/picks.json", encoding="utf-8"))
    if pk.get("dev_output_sha") != verified_dev()[2]: raise seal.Refused("picks.json was not made from the current, verified development output")
    picks = pk["picks"]
    y = np.array([r["correct_em"] for r in rows]); n = len(rows)
    S = {"p_out": np.array([r["p_out"] for r in rows]), "said": said_scores(rows)}
    for rd in config.READERS:
        p = picks[rd]; S[rd] = p["direction"] * signal(rows, rd, p["layer"], p["measure"])
    S2 = {f"{rd}@L{config.FIXED_LAYER}{config.FIXED_MEASURE}": signal(rows, rd, config.FIXED_LAYER, config.FIXED_MEASURE)
          for rd in config.READERS}
    rng = np.random.default_rng(config.SEED); idx = rng.integers(0, n, size=(config.N_BOOT, n))
    def interval(f, level):
        v = np.array([f(i) for i in idx]); bad = int(np.isnan(v).sum()); v = v[~np.isnan(v)]
        lo, hi = np.percentile(v, [(100 - level) / 2, 100 - (100 - level) / 2])
        return lo, hi, bad
    print(f"seal {seal_sha[:16]} | final set: {n} questions | exact match: {int(y.sum())} correct, {int((~y).sum())} wrong "
          f"| said invalid: {sum(r['said'] is None for r in rows)} "
          f"| first token agrees with a plain forward pass on {sum(r['first_id'] == r['fwd_top'] for r in rows)}")
    print("picks:", json.dumps(picks))
    print("\nAUROC with 95% bootstrap interval (label: exact match)")
    for k, s in {**S, **S2}.items():
        lo, hi, bad = interval(lambda i, s=s: auroc(s[i], y[i]), 95.0)
        print(f"  {k:12s} {auroc(s, y):.3f}  [{lo:.3f}, {hi:.3f}]" + (f"  ({bad} resamples had one class)" if bad else ""))
    def compare(a, b, level, SS=S):
        d = auroc(SS[a], y) - auroc(SS[b], y)
        lo, hi, _ = interval(lambda i: auroc(SS[a][i], y[i]) - auroc(SS[b][i], y[i]), level)
        raw_equal = bool(np.array_equal(SS[a], SS[b])); rank_equal = bool(np.array_equal(ranks(SS[a]), ranks(SS[b])))
        note = " | raw values equal" if raw_equal else " | same ranking, different values" if rank_equal else ""
        dr, sz = verdict(lo, hi)
        print(f"  A={a:12s} B={b:12s} diff {d:+.3f}  [{lo:+.3f}, {hi:+.3f}] ({level:.2f}%)  {dr}; {sz}{note}")
    print(f"\nPRIMARY: J, R, J++ against the logit lens, picked signals, 98.33% intervals (margin {config.MARGIN})")
    for rd in config.READERS[1:]: compare(rd, "logit", 100 - 5 / 3)
    print("\nSECONDARY, descriptive (95% intervals)")
    for b in ("p_out", "said"):
        for rd in config.READERS: compare(rd, b, 95.0)
    print("  robustness, no selection: each reader at the fixed layer and measure against the logit lens there")
    fx = lambda rd: f"{rd}@L{config.FIXED_LAYER}{config.FIXED_MEASURE}"
    for rd in config.READERS[1:]: compare(fx(rd), fx("logit"), 95.0, SS=S2)
    valid = np.array([r["said"] is not None for r in rows])
    print(f"\nSENSITIVITY 1, rows with a valid stated confidence only (n={int(valid.sum())}): AUROC of every signal")
    if 0 < y[valid].sum() < valid.sum():
        print("  " + " | ".join(f"{k} {auroc(s[valid], y[valid]):.3f}" for k, s in S.items()))
    else: print("  one class only")
    yc = np.array([r["correct_contain"] for r in rows])
    print(f"SENSITIVITY 2, containment label instead of exact match ({int(yc.sum())} correct; labels disagree on {int((y != yc).sum())}): AUROC of every signal")
    print("  " + " | ".join(f"{k} {auroc(s, yc):.3f}" for k, s in S.items()))

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "select": select()
    elif mode == "verify-dev":
        r, st, sh = verified_dev(); pk = json.load(open(config.BASE + "out/picks.json", encoding="utf-8"))
        if pk.get("dev_output_sha") != sh or pk.get("inference_stamp") != st: raise seal.Refused("picks.json does not match the verified development output")
        print(f"development output verified: {len(r)} rows, stamp {st[:16]}, sha256 {sh}; picks.json was made from it")
    elif mode == "final": final()
    else: sys.exit("usage: analyze.py select|verify-dev|final")
