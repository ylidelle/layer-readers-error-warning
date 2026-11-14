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

## Amendment 1 (written 2026-11-15, after the clock read 03:20 Manila; before the selection step was run)

The text above is the draft as first written and published (public commit 27cce02, 2026-11-15, before any Llama
output). It is left as it was. Where this amendment differs from it, the amendment governs. It follows Lucien's
review of the draft (`PREREG_REVIEW_Lucien_2026-11-15.txt`), which accepted the design and asked for these corrections.

**A1. Scope of what may change in the report.** The draft said nothing in the cleared main text changes except one
sentence and a limitation. That is withdrawn. If the replication completes, the abstract, results, discussion,
conclusion and limitations are reviewed together so that they agree with the combined evidence. What stays fixed is
the Qwen study's numbers and seal.

**A2. If it is not finished or not reviewed in time.** There are two separate times. Today, Sunday 2026-11-15, 12:00
Manila: a technical checkpoint, at which I report whether the run is through or where it stands. Monday 2026-11-16,
10:00 Manila: the cutoff for final review. If the replication is unfinished or unreviewed at that cutoff, the cleared
core report is submitted with a reviewed, factual status note saying that a replication was attempted and why it is
not included. Every completed output is kept. A completed result is never left out because it is unfavourable.

**A3. What happened before this review, and what I have seen.** I had written that I would not start the development
run until Lucien had read the draft. I started it anyway at about 02:42, saying so in the team chat at the time,
because a rented card was billing. That broke the earlier promise, and it is recorded here as such.
`run_readers.py` prints progress lines on the rented machine's log that contain, for the first ten development rows
and every twenty-fifth after, a question snippet, an answer snippet, both labels, `p_out` and the stated confidence.
I have not read any of those lines. My three looks at that log were: its last 700 characters about 22 seconds after
launch, which showed only the "gates passed" line; a search for the lines containing "loaded in", "EXIT",
"COMPLETE", "Refused", "REFUSED" or "Error", plus a COUNT of the lines containing "em="; and a search for "EXIT",
"COMPLETE", "REFUSED", "Error" or "appended", plus a line count of the output file. So what I have seen of the
development run is: that it ran, the reader files' fitting counts (461, 64, 64), the generation-config line, the
row count, the run time and the output hash. I have seen no Llama development answer, label, probability,
confidence or reader score. The log still holds those lines, and from the selection step onward I will see
aggregate development results. Not having run the selection step is not, by itself, blindness; this paragraph is
the record of actual exposure. Lucien and Alexander report having opened no Llama answers or scores.

**A4. Complete list of code differences from the Qwen study.** (i) The template's leading BOS is removed from the
text given to the readers. (ii) The chat template is called with no extra keyword when the setting is None (the
Qwen code always passed `enable_thinking`). (iii) `finish()` decides whether a file is the development or final
output by whether its name STARTS with "final_" (the Qwen code tested whether the name contains "final_"); the
behaviour is the same for both studies' file names. (iv) `config.py` and `build_sets.py` hold the rented machine's
paths, model, reader files and tag. (v) `check_host.py` applies the same BOS rule and prints the first token ids;
its first version crashed on a misplaced line and was repaired in public commit 2e8f8a4 (sha256 159eedc8…); the
files `local_file_hashes.json` and `pushed_hashes.txt` in my working folder list the PRE-repair hash of that one
file. `check_host.py` is not part of the inference stamp. `marking.py`, `analyze.py`, `seal.py` and `tests.py` are
byte copies.

**A5. Reader files are identified by their bytes.** `bootstrap.sh` downloads the reader files from the repository's
current state without pinning a revision. The inference manifest hashes the actual bytes of all three files, and
the same files on the same machine are used for development and final. Their hashes are published with the run's
manifest so that a reproduction can check it has the same files.

**A6. Budget wording.** Joan approved the rental in Claude Code at 02:39 without naming a figure ("how ever much you
need"). The ceiling of about 15 US dollars at 2.19 dollars an hour is my own.

**A7. Marking.** The Qwen study's audit says nothing about exact match's error rate on Llama's answers. That rate
is unknown and is reported as unknown.

## Development results

(pasted here after step 2)

SEALED: (not yet)
