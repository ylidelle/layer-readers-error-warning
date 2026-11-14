# Pre-registration: which reader file best tells that Qwen3.5-4B is about to be wrong?

Author of this document and of the code: Opie. Reviewers before sealing: Alexander and Lucien.
All times are Manila time on 2026-11-14, read from the clock or from file times, not remembered.

## History of this document (so nothing is called "fixed in advance" that was not)

- 18:46: question sets built by `build_sets.py` (file times of `data/dev.jsonl`, `data/final.jsonl`).
- 18:46 to 18:59: a first development run with the first version of the inference script. It was stopped at 167 of
  300 rows. Those rows are kept in `archive_pre_review/` and are used for NOTHING in this study. I saw their
  right/wrong marks and example answers scroll past; I looked at no reader score from them.
- about 18:49: first draft of this document (kept: `archive_pre_review/PREREG_draft_before_Lucien_review.md`,
  which already includes Alexander's amendments of about 18:52 to 18:54).
- 18:58: Lucien's review, "revise before sealing" (`PREREG_REVIEW_Lucien_2026-11-14.txt`).
- 19:00 to 19:05: inference, marking, sealing and analysis code rewritten; synthetic tests and the host check run.
- 19:13: Lucien's recheck; the development-provenance gate and three wordings added (no stamped file touched).
- 19:05 to 19:25: development run under the rewritten pipeline (300 rows, output sha256 3f1d6ce5…eca2c2).
- 19:25: blind audit packets written. 19:28 and 19:34: Alexander's and Lucien's hashes posted. 19:34: audit
  comparison, then the selection step, in that order.
- The final set: not run, not opened, not printed by any script, at any point before the seal.

So: the question sets, the model, the reader files and the prompt were fixed before any model saw any question.
The label rule, the confidence parser, the verdict rule and the gates were fixed AFTER review and after I had seen
the stopped run's marks, but BEFORE any development reader score was looked at and before any final-set work.

## Question

Before a model writes its answer, four "readers" can be pointed at its internal state: the logit lens, the J-lens,
the R-lens and J++. Which of them best separates the questions the model will answer correctly from the ones it
will answer wrongly?

The claim we can support is narrow: **which of these ready-made reader FILES best predicts THIS model's errors on
THIS kind of question.** The files were fitted by other people on different amounts of text (J on 417 passages; R
and J++ on 64 each), so a difference between files is not a difference between methods.

## Fixed materials (their hashes are bound by SEAL.json, not just named here)

- Model: Qwen/Qwen3.5-4B, local copy of revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, chat template, thinking
  off, bfloat16, greedy decoding. The model ships no generation config. On the three made-up host-check prompts the
  processed generation scores equalled the raw generation logits; that is evidence from three prompts, not a proof
  for every prompt. A plain forward pass differed slightly from generation (largest difference 0.03 to 0.06 in a
  logit), so not every computation path gives bit-identical numbers.
- Reader files: Neuronpedia copies for qwen3.5-4b (J-lens, R-lens, J++), used as published; logit lens = the same
  library call with the Jacobian switched off. Library: jlens 0.1.0 as installed.
- Layers: the model has 32 blocks (0 to 31). The reader files cover 0 to 30. All four readers are read at 0 to 30
  only. Block 31, where the logit lens equalled the model's output on the host-check prompts, is read by none.
  That removes a built-in duplication; a reader at block 30 can still happen to rank questions like `p_out`.
- Questions: TriviaQA, rc.nocontext, validation split (file sha256 48a5005c…0d5b201). Rule in `build_sets.py`:
  one row per question id; drop any question whose answer or any listed alias contains a digit; shuffle with seed
  20261114; first 300 = development, next 1000 = final.
- Prompt: the question plus " Answer with just the answer, no explanation." At most 24 new tokens.
- Length policy: one limit (4096 tokens) for readers and generation; a longer prompt is an error, never truncated.
  On every row the script requires the token ids seen by all four reader calls and by generation to be identical.

## Label

**Primary: exact match.** An answer is correct if, after TriviaQA's own normalisation (lower case; punctuation to
spaces; a/an/the removed; single spaces), it EQUALS one of the accepted aliases. Known cost: a correct answer
wrapped in extra words is marked wrong.
**Secondary, sensitivity only: containment.** An accepted alias appears in the answer as whole words. This is a
custom, looser rule (it accepts "Not Paris; London" for Paris) and is never called "correctness" without that name.

Marking audit (development rows only, before sealing): `make_audit.py` draws 100 of the 300 development questions
with a fixed seed. Two reviewers each get the question, the accepted aliases and the model's answer, with no
automatic label and no score, and judge correct / incorrect / uncertain with a short reason, working apart, by a rubric they agreed before
opening the packet (meaning decides; a negated answer or unresolved alternatives are not rescued by containing an
alias; a right answer under a missing alias counts only with a source; a doubtful reference is flagged, not
overruled). Each hashes their own file before comparing. We report how often they agree and,
on the rows where both are sure and agree, how often each label rule differs from them. One hundred rows cannot
give a precise error rate and we will not claim one. The audit can lead to a change of label rule only before
sealing; after sealing no manual judgement replaces the primary label, and any corrected analysis is reported
separately beside the registered one.

## Signals (higher is always oriented to mean "more likely correct")

- Readers, read at the last prompt position before the model writes: for each reader and each layer 0 to 30, top-1
  probability, entropy, top-1 minus top-2 logit. None uses the answer key or the model's answer.
- `p_out`: the probability of the model's own first answer token, from generation's processed scores (the raw-logit
  version is stored beside it). Read at the same moment as the readers. Direction fixed: higher = more likely correct.
- `said`: the model is asked, AFTER it has answered, "How confident are you…? Reply with a number from 0 to 100
  only." The WHOLE reply must be one number from 0 to 100 (decimals and a % sign allowed); anything else is
  invalid. Direction fixed: higher = more likely correct; an invalid reply ranks below every valid one. `said` has
  seen the answer; the readers and `p_out` have not. It is a later signal with more information.

## Chosen on the development set only

`analyze.py select` requires a development run recorded as COMPLETE (300 rows) under the current inference stamp,
whose output file still has the hash recorded at completion. That hash is written into picks.json, and the audit
packet is made from the same verified file. For each reader it
scores every layer, measure and direction and keeps the highest development AUROC (label: exact match). Ties: lower
layer, then measure in the order p1, ent, margin; an AUROC of exactly 0.5 counts as the positive direction. Same
procedure for all four readers.

DEVELOPMENT RESULTS AND PICKS (output of `analyze.py select`, run once, at 19:34, AFTER the audit comparison):

```
development set: 300 questions | exact match: 111 correct, 189 wrong | containment label: 123 correct | the two labels disagree on 12
said: 0 invalid replies; distinct valid values [0.0, 95.0, 100.0]
baselines on dev: p_out 0.865 | said 0.566
first token agrees with a plain forward pass on 300 of 300
logit pick: layer 30 ent    direction -1 dev AUROC 0.702 | next: L28 ent 0.699, L30 p1 0.694, L27 ent 0.685 | fixed L23 p1: 0.539
J     pick: layer 30 ent    direction -1 dev AUROC 0.826 | next: L30 p1 0.797, L29 ent 0.744, L29 p1 0.744 | fixed L23 p1: 0.626
R     pick: layer 30 ent    direction -1 dev AUROC 0.773 | next: L30 p1 0.763, L29 ent 0.717, L29 p1 0.715 | fixed L23 p1: 0.631
J++   pick: layer 30 ent    direction -1 dev AUROC 0.752 | next: L30 p1 0.745, L29 ent 0.705, L28 ent 0.695 | fixed L23 p1: 0.623
picks.json written under stamp b4a9d9264453af83 | sha256 462c9b63fe099a4b789f2efd423cb953f87d65da47e75c1c9f6cdeb18cf1051c
```

In words: every reader's pick is layer 30, entropy, with LOWER entropy meaning more likely correct. On development
data the order is J-lens, R-lens, J++, logit lens, and all four are below `p_out`. These are development numbers,
flattered by selection; they are stated here so that the final result can be compared with them, not as findings.

MARKING AUDIT (output of `make_audit.py compare`, 19:34; reviewer A = Lucien, reviewer B = Alexander; each scored
alone and posted the hash of their file in Kwentuhan before the comparison; the hashes below match their posts):

```
reviewer A ratings file sha256 c08985573e2a95e6a82005207852968ceb0189e48168a9a9a028bd52bb4ef4c8  (each reviewer confirms this is the hash they recorded)
reviewer B ratings file sha256 e851410b6e59ae4be24082a508361fe5a91554a829fbb026166b1d8acae8a35d  (each reviewer confirms this is the hash they recorded)
ALL 100 ROWS: same judgement on 93; at least one 'uncertain' on 7; different judgements on 7
DETERMINATE-AGREEMENT SUBSET: 93 of 100 rows where both reviewers gave the same sure judgement. Marker errors below use THIS denominator; it is not an unbiased estimate for all 100.
correct_em: marked correct but judged incorrect: 0 of 93; marked wrong but judged correct: 6 of 93
   ids marked correct, judged incorrect: []
   ids marked wrong, judged correct: ['tb_53', 'qw_8018', 'sfq_16268', 'qw_3808', 'odql_1018', 'qg_2879']
correct_contain: marked correct but judged incorrect: 1 of 93; marked wrong but judged correct: 4 of 93
   ids marked correct, judged incorrect: ['sfq_14830']
   ids marked wrong, judged correct: ['qw_8018', 'sfq_16268', 'qw_3808', 'qg_2879']
rows without determinate agreement: [('qg_4484', 'uncertain', 'incorrect'), ('odql_4233', 'incorrect', 'uncertain'), ('bb_8661', 'uncertain', 'correct'), ('bb_118', 'uncertain', 'incorrect'), ('bb_3702', 'uncertain', 'incorrect'), ('qw_12970', 'incorrect', 'uncertain'), ('dpql_3845', 'uncertain', 'incorrect')]
```

In words: on the 93 rows where both reviewers were sure and agreed, exact match never marked a wrong answer
correct, and marked 6 correct answers wrong. Containment marked 1 wrong answer correct and 4 correct answers wrong.
Decision, made by me before running the selection step and with no reader score seen: the label rule stays as
registered (exact match primary). Its error runs one way: some correct answers are counted as wrong. Ninety-three
rows cannot give a precise rate, and the 7 rows without sure agreement are not in that count.

Limits of the audit, disclosed by Lucien (Kwentuhan, 19:34): before he opened his packet, Alexander's post showed
Alexander's totals (36 / 62 / 2) and his verdict on one item (the "portamento" row, marked correct with a cited
source). Lucien did not open Alexander's file, and checked that item himself against a dictionary, but that item is
not fully blind and the totals were visible. The audit is therefore two separate ratings, not two perfectly
independent ones. No adjudication of the 7 unsettled rows was done; both original rating files are kept unchanged
(Lucien's sealed copy and receipt: `out/audit_reviewer_A_Lucien_SEALED_20261114.json`, `…_receipt_20261114.json`).

Because each pick is the best of 186 tries on 300 questions, its development score flatters it. Only final-set
scores count. As a check that selection is not doing the work, the final analysis also reports every reader at one
setting fixed without data: layer 23, top-1 probability, positive direction.

## Final analysis, run once

Orientation: AUROC is the probability that a randomly chosen correct answer gets a higher score than a randomly
chosen wrong one (ties count half); 0.5 is chance.

**Primary question: does J, R or J++ differ from the logit lens?** Three paired comparisons on the same 1000
questions, each a difference in AUROC with a bootstrap interval (10,000 resamples of questions, seed 20261114).
Because there are three, each interval is 98.33% (Bonferroni). The PICKED signals carry the conclusion.

**Each comparison gets two separate statements from that one interval** (A minus B; ends exactly on 0 or on the
margin leave the statement unresolved):
- Direction: "A higher" (whole interval above 0), "B higher" (whole interval below 0), or "unresolved".
- Practical size against the margin of 0.05: "practically equivalent within the margin" (whole interval inside
  -0.05 to +0.05), "better by more than the margin" (whole interval beyond 0.05 on one side), or "unresolved".
An interval such as [0.01, 0.08] therefore reads: A higher; practical size unresolved. We will say "a practically
meaningful improvement" only when the whole interval is beyond 0.05. "Practically equivalent within the margin" is
not a claim that there is no difference at all.

**Why 0.05:** it is our judgement of the smallest difference in AUROC that would matter in practice, made before
any final-set model output was generated or inspected (the final question file already existed). It is a judgement, not a measured quantity, and a 0.05 change in AUROC is not shown to
change any real decision.

**Secondary, descriptive only (95% intervals, no verdict language in the abstract):** each reader against `p_out`;
each reader against `said`; each reader against the logit lens at the fixed layer-23 setting. Two sensitivity
tables: every signal on the rows with a valid `said` only; every signal under the containment label.

If two compared signals turn out to have equal raw values, or the same ranking with different values, the script
says which. Nothing is called identical "by construction" on the strength of that alone.

## Gates (in code, tested on synthetic fixtures in `tests.py`, with asserts on and off)

- `run_readers.py final` refuses before reading any question or loading the model unless SEAL.json exists and the
  current state matches it in every entry: settings; inference, marking, sealing and analysis code; both question
  files; the three reader files; every model file; package versions; this document; the picks.
- Final rows are stamped with the seal's hash. Resume is allowed only into rows with that stamp that are a
  duplicate-free prefix of the expected order. A torn last line after a crash is an error unless the run is
  restarted with `--drop-torn-tail`, which removes that one line and logs it.
- When the final file is complete its hash is recorded; `analyze.py final` refuses if the file has changed since,
  or if the rows are not exactly the 1000 expected ids once each, or any field the analysis uses (both labels,
  `p_out`, `said`, every reader's three measures at every layer) is missing or not finite, or only one outcome
  class is present. Diagnostic fields (raw-logit probability, token counts, token-id hashes) are stored but not
  validated.
- The final run may be repeated only after a crash, under the same seal, and the repeat is reported.

## What we will not do

- Swap, drop or re-mark final-set questions after seeing results.
- Present any analysis that uses the answer key or the model's answer to define a signal as a predictor.
- Read a result about these files as a result about the methods behind them.
- Run a second model under this document. A Llama 3.1 8B run, if there is time, gets its own short registration
  before its own evaluation, is labelled a secondary replication, and does not delay this result.

## Known limits, stated before the result

One small model. One question set (English trivia, no digit answers). Reader files fitted on unequal amounts of
text by other people. Exact match marks some correct answers wrong. Readers are read at one position only. The
bootstrap interval is an approximation. The development picks come from 300 questions.

Reviews taken: Alexander (`From Alexander/SPRINT_PREREG_REVIEW_Alexander_2026-11-14.md`, seven points);
Lucien (`PREREG_REVIEW_Lucien_2026-11-14.txt`, six points and the small corrections). Lucien's recheck (`PREREG_RECHECK_Lucien_2026-11-14.txt`):
six repairs confirmed; one more gate asked for (development output must match its completion hash before selection
and audit) and three wordings; all taken.

SEALED: 2026-11-14 20:00 Manila, by Opie, after the room was given from 19:50 to 20:00 to object and nobody did. The hash of
this file and of SEAL.json are posted in Kwentuhan. Nothing in this file changes after this line was written.
