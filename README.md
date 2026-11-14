# Does a better layer reader see mistakes coming?

Code, sealed pre-registration, reviews and outputs for our entry to the Apart Research AI x Epistemics Research
Sprint (13 to 15 November 2026). Team: Claude Orion "Opie" Bennett, Claude Alexander Bennett, Lucien Vale,
Joan Miranda. Three of the four are AI systems.

**Question.** Before Qwen3.5-4B writes an answer to a closed-book trivia question, can a reader pointed at one of
its layers tell that the answer will be wrong, and do the newer readers (J-lens, R-lens, J++) do this better than
the logit lens?

**Result, in one paragraph.** On 1,000 held-out questions the J-lens file scored higher than the logit lens
(AUROC 0.817 against 0.748; difference +0.069, 98.33% interval +0.041 to +0.097); the interval crosses our
pre-set margin of 0.05, so the practical size of the gain is unresolved. The R-lens was higher by less than the
margin, on an interval within a hair of both of its lines. J++ was practically equivalent to the logit lens. Every
reader was below the model's own first-token output probability (0.863). Stated confidence was near chance (0.583).
The full output is `out/final_analysis.log`.

## What was made BEFORE the sprint (preexisting work)

- Downloads of the model (Qwen/Qwen3.5-4B, revision 851bf6e8) and of the three Neuronpedia reader files, and load
  tests of them on our hardware (10 to 12 November 2026). Those test scripts are not in this folder.
- Reading notes and a prior-art search (10 to 11 November 2026).
- The reader library (`jlens` 0.1.0, Anthropic, Apache-2.0) and the reader files are other people's work, used
  unchanged.

Everything in this folder was written inside the sprint window, on 14 November 2026 (Manila time).

## Files

| File | What it is |
|---|---|
| `PREREG.md`, `SEAL.json` | The sealed pre-registration and the seal (hashes of code, data, reader files, model files, picks). |
| `config.py`, `seal.py`, `marking.py`, `run_readers.py` | Settings, gates, marking and the inference run. A change to any of these changes the inference stamp. |
| `analyze.py` | Development selection and the final analysis. |
| `build_sets.py` | Builds the development and final question sets from TriviaQA by a fixed rule. |
| `tests.py` | 115 synthetic checks of marking, parsing, verdicts, AUROC and every gate. No real data, no model. |
| `check_host.py` | Token-id and output-probability checks on made-up prompts (needs the GPU). |
| `make_audit.py` | The blind marking-audit packet and its comparison. |
| `make_results.py`, `precision_check.py` | Report tables and figures from the analysis log; the three primary intervals to six decimals. |
| (reviews) | Lucien's four review documents and his clearance are his own files and are not in this first release; they will be added if he agrees. Their findings are summarised in `PREREG.md` and `REPORT_NOTES_post_seal.md`. |
| `REPORT_NOTES_post_seal.md` | Corrections made after the seal, including a process fault of ours. Read this beside `PREREG.md`. |
| `report/` | The report as cleared by the reviewer: `PAPER_EPI.pdf`, `PAPER_EPI.md`. |
| `out/` | Run outputs (question ids, answers and signals; no question text), manifests, analysis logs, figures. |

## To reproduce

1. Put the model and the three reader files where `config.py` points, and TriviaQA's `rc.nocontext` validation
   parquet where `build_sets.py` points. Run `build_sets.py`; the two hashes must match `config.py`.
2. `python tests.py` and `python -O tests.py`.
3. `python run_readers.py dev`, then `python analyze.py select`.
4. Our `SEAL.json` binds our exact files, package versions and outputs, so a fresh copy will not match it. To rerun
   the final set you make your own seal (`seal.py make`) and then `run_readers.py final` and `analyze.py final`.

The marking audit packets contain TriviaQA questions and model answers from the development set only. No personal
data is used anywhere.

## Not included in this release

The two question files (`data/dev.jsonl`, `data/final.jsonl`) and the audit packets contain TriviaQA questions and are not
redistributed here; `build_sets.py` rebuilds the question files exactly (their hashes are in `config.py`), and
`make_audit.py` rebuilds the packet from them. The archived first scripts and the stopped first run are also left out.
Because the question files are absent, `seal.py check` will not pass on a bare checkout until they are rebuilt.
