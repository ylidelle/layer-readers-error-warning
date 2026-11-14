**Table 1.** AUROC for telling correct from wrong answers on the 1,000 final questions (0.5 is chance), with 95% bootstrap intervals. Reader settings were chosen on the development set.

| Signal | AUROC | 95% interval |
|---|---|---|
| Output probability | 0.863 | 0.840 to 0.886 |
| J-lens | 0.817 | 0.790 to 0.843 |
| R-lens | 0.773 | 0.743 to 0.803 |
| J++ | 0.757 | 0.726 to 0.788 |
| Logit lens | 0.748 | 0.717 to 0.779 |
| Stated confidence | 0.583 | 0.568 to 0.599 |

**Table 2.** Primary comparisons: each newer reader minus the logit lens, difference in AUROC on the same questions, 98.33% intervals.

| Comparison | Difference | 98.33% interval | Direction | Practical size (margin 0.05) |
|---|---|---|---|---|
| J-lens minus logit lens | +0.069 | +0.041 to +0.097 | J-lens higher | practical size unresolved |
| R-lens minus logit lens | +0.025 | +0.001 to +0.050 | R-lens higher | practically equivalent within the margin |
| J++ minus logit lens | +0.009 | -0.017 to +0.036 | direction unresolved | practically equivalent within the margin |

**Table 3.** Secondary, descriptive: each reader minus a baseline, 95% intervals.

| Comparison | Difference | 95% interval |
|---|---|---|
| Logit lens minus output probability | -0.115 | -0.145 to -0.085 |
| J-lens minus output probability | -0.046 | -0.067 to -0.026 |
| R-lens minus output probability | -0.090 | -0.115 to -0.065 |
| J++ minus output probability | -0.106 | -0.133 to -0.079 |
| Logit lens minus stated confidence | +0.165 | +0.130 to +0.199 |
| J-lens minus stated confidence | +0.234 | +0.204 to +0.263 |
| R-lens minus stated confidence | +0.190 | +0.157 to +0.223 |
| J++ minus stated confidence | +0.174 | +0.140 to +0.208 |

**Table 4.** Robustness, no selection: every reader at layer 23, top-1 probability.

| Signal | AUROC | 95% interval |
|---|---|---|
| Logit lens at layer 23 | 0.559 | 0.523 to 0.595 |
| J-lens at layer 23 | 0.634 | 0.598 to 0.668 |
| R-lens at layer 23 | 0.639 | 0.604 to 0.673 |
| J++ at layer 23 | 0.645 | 0.610 to 0.678 |
| J-lens minus logit lens, both at layer 23 | +0.075 | +0.032 to +0.117 |
| R-lens minus logit lens, both at layer 23 | +0.080 | +0.043 to +0.116 |
| J++ minus logit lens, both at layer 23 | +0.086 | +0.049 to +0.123 |
