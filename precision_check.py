"""Post-result, 2026-11-14 (asked for by Alexander): the registered analysis prints three decimals, and the R-lens
interval's upper end printed as +0.050, exactly the margin. This script recomputes the SAME three primary intervals
(same rows, same seed, same resamples, same functions from analyze.py) and prints six decimals. It changes nothing
and tests nothing new. The verdict rule in analyze.verdict uses strict inequalities (lo > -margin and hi < margin)."""
import sys, json
import numpy as np
import config, seal, analyze
sys.stdout.reconfigure(encoding="utf-8")
seal_sha = seal.require_sealed()
rows = seal.load_complete("final", seal_sha)
picks = json.load(open(config.BASE + "out/picks.json", encoding="utf-8"))["picks"]
y = np.array([r["correct_em"] for r in rows]); n = len(rows)
S = {rd: picks[rd]["direction"] * analyze.signal(rows, rd, picks[rd]["layer"], picks[rd]["measure"]) for rd in config.READERS}
idx = np.random.default_rng(config.SEED).integers(0, n, size=(config.N_BOOT, n))
level = 100 - 5 / 3
for rd in config.READERS[1:]:
    v = np.array([analyze.auroc(S[rd][i], y[i]) - analyze.auroc(S["logit"][i], y[i]) for i in idx])
    lo, hi = np.percentile(v, [(100 - level) / 2, 100 - (100 - level) / 2])
    d = analyze.auroc(S[rd], y) - analyze.auroc(S["logit"], y)
    print(f"{rd:4s} minus logit: diff {d:+.6f}  [{lo:+.6f}, {hi:+.6f}]  hi < 0.05: {bool(hi < config.MARGIN)}  lo > 0.05: {bool(lo > config.MARGIN)}  verdict {analyze.verdict(lo, hi)}", flush=True)
