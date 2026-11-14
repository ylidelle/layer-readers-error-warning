"""SPRINT report helper (written 2026-11-14 while the final run was still going; reads final rows only when run,
AFTER `analyze.py final`). Numbers for the report's tables are PARSED from out/final_analysis.log, never retyped.
Also draws two figures. Figure 2 (AUROC by layer) is DESCRIPTIVE and was not pre-registered."""
import sys, re, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config, seal, analyze
sys.stdout.reconfigure(encoding="utf-8")
B = config.BASE
log = open(B + "out/final_analysis.log", encoding="utf-8").read()
NAMES = {"p_out": "Output probability", "said": "Stated confidence", "logit": "Logit lens", "J": "J-lens", "R": "R-lens", "J++": "J++"}
au = {}
for m in re.finditer(r"^\s+(\S+)\s+(\d\.\d{3})\s+\[(\d\.\d{3}), (\d\.\d{3})\]", log, re.M):
    au[m.group(1)] = tuple(float(x) for x in m.groups()[1:])
cmp_rows = re.findall(r"A=(\S+)\s+B=(\S+)\s+diff ([+-]\d\.\d{3})\s+\[([+-]\d\.\d{3}), ([+-]\d\.\d{3})\] \((\d+\.\d+)%\)\s+(.*)", log)
md = ["**Table 1.** AUROC for telling correct from wrong answers on the 1,000 final questions (0.5 is chance), with 95% bootstrap intervals. Reader settings were chosen on the development set.", "",
      "| Signal | AUROC | 95% interval |", "|---|---|---|"]
for k in ("p_out", "J", "R", "J++", "logit", "said"):
    a, lo, hi = au[k]; md.append(f"| {NAMES[k]} | {a:.3f} | {lo:.3f} to {hi:.3f} |")
md += ["", "**Table 2.** Primary comparisons: each newer reader minus the logit lens, difference in AUROC on the same questions, 98.33% intervals.", "",
       "| Comparison | Difference | 98.33% interval | Direction | Practical size (margin 0.05) |", "|---|---|---|---|---|"]
for a, b, d, lo, hi, lvl, v in cmp_rows:
    if b == "logit" and "@" not in a:
        dr, sz = [x.strip() for x in v.split("|")[0].split(";")]
        md.append(f"| {NAMES[a]} minus logit lens | {d} | {lo} to {hi} | {dr.replace('A', NAMES[a]).replace('B', 'logit lens')} | {sz.replace('A ', NAMES[a] + ' ').replace('B ', 'logit lens ')} |")
md += ["", "**Table 3.** Secondary, descriptive: each reader minus a baseline, 95% intervals.", "", "| Comparison | Difference | 95% interval |", "|---|---|---|"]
for a, b, d, lo, hi, lvl, v in cmp_rows:
    if b in ("p_out", "said"): md.append(f"| {NAMES[a]} minus {NAMES[b].lower()} | {d} | {lo} to {hi} |")
md += ["", "**Table 4.** Robustness, no selection: every reader at layer 23, top-1 probability.", "", "| Signal | AUROC | 95% interval |", "|---|---|---|"]
for k, (a, lo, hi) in au.items():
    if "@" in k: md.append(f"| {NAMES[k.split('@')[0]]} at layer 23 | {a:.3f} | {lo:.3f} to {hi:.3f} |")
for a, b, d, lo, hi, lvl, v in cmp_rows:
    if "@" in a: md.append(f"| {NAMES[a.split('@')[0]]} minus logit lens, both at layer 23 | {d} | {lo} to {hi} |")
open(B + "out/results_tables.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
print("\n".join(md))
# Figure 1: AUROC with intervals
order = ["p_out", "J", "R", "J++", "logit", "said"]
fig, ax = plt.subplots(figsize=(6.2, 3.2), dpi=200)
for i, k in enumerate(order):
    a, lo, hi = au[k]
    ax.errorbar([a], [len(order) - 1 - i], xerr=[[a - lo], [hi - a]], fmt="o", color="#444" if k in ("p_out", "said") else "#b8860b", capsize=3)
ax.set_yticks(range(len(order))); ax.set_yticklabels([NAMES[k] for k in order][::-1])
ax.axvline(0.5, color="#999", lw=0.8, ls="--"); ax.set_xlabel("AUROC on 1,000 final questions (0.5 = chance)"); ax.set_xlim(0.45, 1.0)
fig.tight_layout(); fig.savefig(B + "out/fig1_auroc.png"); plt.close(fig)
# Figure 2 (descriptive): AUROC of low entropy by layer, development and final
seal_sha = seal.require_sealed()
sets = {"development (300)": analyze.verified_dev()[0], "final (1,000)": seal.load_complete("final", seal_sha)}
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), dpi=200, sharey=True)
desc = {}
for ax, (title, rows) in zip(axes, sets.items()):
    y = np.array([r["correct_em"] for r in rows])
    for rd, col in zip(config.READERS, ["#444444", "#b8860b", "#2a7f62", "#8a3ffc"]):
        curve = [analyze.auroc(-analyze.signal(rows, rd, L, "ent"), y) for L in config.LAYERS]
        desc[f"{title}|{rd}"] = [round(c, 4) for c in curve]
        ax.plot(config.LAYERS, curve, label=NAMES[rd], color=col, lw=1.4)
    ax.axhline(analyze.auroc([r["p_out"] for r in rows], y), color="#999", ls=":", lw=1); ax.axhline(0.5, color="#ccc", lw=0.8, ls="--")
    ax.set_title(title, fontsize=9); ax.set_xlabel("layer")
axes[0].set_ylabel("AUROC of low entropy"); axes[1].legend(fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(B + "out/fig2_by_layer.png"); plt.close(fig)
json.dump(desc, open(B + "out/fig2_by_layer.json", "w"), indent=1)
print("figures written: out/fig1_auroc.png, out/fig2_by_layer.png (dotted line = output probability)")
