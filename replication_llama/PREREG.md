# Registration of a secondary replication on Llama 3.1 8B

Author: Opie. Written on 2026-11-15 (Manila), AFTER the results of the primary study on Qwen3.5-4B were seen by all
four of us. It is prospective for this model only: no Llama output of any kind existed when it was written, apart
from a three-prompt load test on 2026-11-12 that scored nothing. It is not independent of the primary results, and
it reuses the same questions, so the two studies are not independent samples.

The primary study's sealed plan (`../epi_sprint/PREREG.md`, sha256 09797bbc…9e2edb) said that a Llama run would get
its own registration before its own evaluation, be labelled a secondary replication, and not delay the primary
result. This is that registration. The primary seal and every primary file stay untouched.

## What is the same as the primary study

Everything not listed under "What is different": the 300 development and 1,000 final questions (same files, same
hashes), the prompt suffix, greedy decoding with at most 24 new tokens, the three reader measures at the last prompt
position for layers 0 to 30, the two baselines and their fixed directions, the whole-reply confidence parser, exact
match as the primary label with containment as a labelled sensitivity check, the selection procedure and its
tie-breaks, the fixed layer-23 top-1 check, AUROC with ties counted half, 10,000 paired bootstrap resamples with seed
20261114, three primary comparisons (J, R, J++ each against the logit lens) with 98.33% intervals, the two-part
verdict against a margin of 0.05, and the gates. `marking.py`, `analyze.py` and `seal.py` are byte copies.

## What is different

- Model: meta-llama/Llama-3.1-8B-Instruct, snapshot 0e9e39f249a16976918f6564b8830bc894c89659, bfloat16, on a rented
  A100. Its chat template is used as shipped, with no extra switch. As shipped it adds a system header with fixed
  dates; we do not change it.
- Reader files: `neuronpedia/jacobian-lens`, folder `llama3.1-8b-it` (J-lens, R-lens, J++), used as published. In the
  load test the J-lens reported 461 fitting prompts and the R-lens and J++ 64 each; the run prints the counts again.
  The model has 32 blocks; the files cover 0 to 30; block 31 is read by no reader. The host check on the rented
  machine must confirm that mapping before anything is scored.
- Tokens: this model's template begins with a BOS token and the reader library adds one of its own, so the BOS is
  removed from the text given to the readers. This is the one code change in `run_readers.py`. The per-row check
  that all four readers and generation see identical token ids is what verifies it; if it fails, the run stops.
- Selection is repeated from scratch on Llama's own development outputs. Qwen's winning layer and measure are NOT
  carried over. One selection, one final run; no alternative approaches are tried.
- No second marking audit. Exact match stays the primary label by rule. This model may phrase answers differently, so
  the share of correct answers that exact match fails may differ from the primary study; we do not know it.

## Order of work, and the gates

1. On the rented machine: rebuild the question files with `build_sets.py` (hashes must equal the registered ones),
   run `tests.py` normally and with `-O`, run `check_host.py`.
2. Development run (300), then `analyze.py select`.
3. The picks and development numbers are pasted into this file under "Development results", the SEALED line is
   written, and the file's hash is posted in our team chat.
4. **Lucien gives an explicit GO in words before the final run.** Silence is not a GO.
5. `seal.py make` on the rented machine, final run (1,000), `analyze.py final` once.
6. Outputs are copied back and their hashes compared with the hashes printed on the rented machine.

## Reporting, fixed now

- **If the final analysis completes, it is reported whatever it shows.** The only fallback is for a run that does not
  complete or does not pass review in time; then the report says that a replication was attempted and why it is
  absent.
- **Stop time: Monday 2026-11-16, 10:00 Manila.** If the replication is not through Lucien's check by then, the
  already cleared report is submitted as it stands.
- It is reported in an appendix, with one sentence in the main text. Nothing in the cleared main text changes
  except that sentence and the matching limitation.
- It is called a secondary replication on a second model family. Qwen 4B and Llama 8B differ in architecture,
  training, tokenizer and fitted reader files at once, so no difference between the two studies is attributed to
  model size, and nothing here is evidence about large models. Results are not pooled across the two models.
- The same wording rules as the primary report: a difference between reader files is not a difference between
  methods; AUROC is discrimination against a noisy label, not calibration or usefulness in deployment.

Conditions adopted from review before this was written: Alexander (report whatever it shows; appendix; stop time)
and Lucien (repeat selection on Llama's own development data; freeze before development results; validate on the
rental machine; explicit GO; disclose that this was written after the Qwen results), Kwentuhan, 2026-11-15.

## Development results

(pasted here after step 2)

SEALED: (not yet)
