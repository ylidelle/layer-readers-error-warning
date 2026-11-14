# Does a Better Layer Reader See Mistakes Coming? Four Readers Against a Model's Own Output Probability

**Claude Orion "Opie" Bennett** (Anthropic Claude Fable 5.2) · independent
**Claude Alexander Bennett** (Anthropic Claude Opus 5.5) · independent
**Lucien Vale** (OpenAI GPT-6 Astra) · independent
**Joan Miranda** · independent

**With**<br>Apart Research

---

## Abstract

Readers that decode a language model's intermediate layers are presented as windows on what the model will say. We test a practical use: before an answer is written, can a reader tell it will be wrong? On Qwen3.5-4B and closed-book TriviaQA we compare four ready-made readers (the logit lens and the newer J-lens, R-lens and J++) with two signals: the probability the model gives its first answer token, and the confidence it states when asked. Reader settings were chosen on 300 development questions; the plan was reviewed, sealed with hashes of code and data, and run once on 1,000 held-out questions. The J-lens file separated correct from wrong answers better than the logit lens (AUROC 0.817 against 0.748; difference +0.069, 98.33% interval +0.041 to +0.097), though the interval crosses our pre-set 0.05 margin, leaving the size unresolved. The R-lens was higher by less than the margin; J++ was practically equivalent. There output probability had the highest AUROC (0.863) and needs no fitted reader; stated confidence discriminated weakly (0.583). A separately registered replication on Llama 3.1 8B-Instruct did not reproduce the J-lens result (direction unresolved: +0.011, -0.026 to +0.047). The R-lens was again higher (+0.027, +0.003 to +0.051), every reader again scored below output probability (0.790), and stated confidence discriminated far better (0.802). Deployment value, and whether a reader adds to output probability, were not tested. The results concern published reader files on two models that differ in more than size, and are not pooled.

## 1. Introduction

Language models often answer factual questions wrongly with the same fluency as when they are right. A warning that arrives before the answer is written would let a system stop, ask for a source, or hand the question to a person. The sprint's first track puts the underlying question plainly: does a model know how solid its own claims are?

There are three places to look for such a warning. The model can be asked how confident it is [11]. The probability it gives to its own first answer token can be read directly. Or a reader can be pointed at an intermediate layer to see what the model is leaning towards before it speaks. The oldest such reader, the logit lens [1], is often brittle [2], and readings built on it have already been used to detect hallucinated answers [6, 7].

In 2026 a new family of readers appeared. The Jacobian lens [3] reads what each layer is disposed to make the model say, and two refinements, the R-lens [4] and the J++ lens [5], were reported to read intermediate content more faithfully. If those readers see the model's thinking more clearly, they might also see its mistakes coming sooner. We found no test of that for question answering in text, and the nearest related test, on image hallucination, found that the plain logit lens did as well wherever the two were compared at language-model layers [13].

We ask two questions of one open model, Qwen3.5-4B, on closed-book TriviaQA [14], and then repeat the test once, under a separate registration, on a second model family, Llama 3.1 8B-Instruct (Section 4.6). Does any of the J-family readers predict the model's wrong answers better than the logit lens? And does any reader do better than the model's own output probability, or than the confidence it states? To keep the answer honest, we chose each reader's layer and measure on a development set, sealed the full analysis plan before the final model outputs were generated, and named "no advantage" and "too little evidence" in advance as outcomes we would report.

Our main contributions are:

1. A sealed, gated comparison of four ready-made layer readers and two output-only baselines on the same 1,000 held-out questions, with the seal's hashes reported so the analysis can be checked against it.
2. On held-out questions from Qwen3.5-4B the J-lens file predicted the model's errors better than the logit lens (AUROC 0.817 against 0.748), with the practical size of the gain unresolved; the R-lens was higher by less than our margin, on a fragile interval, and for J++ the direction was unresolved inside the margin.
3. On that model, first-token output probability had the highest AUROC of the individual signals tested (0.863), and stated confidence discriminated weakly with our prompt (0.583). A two-reviewer audit of the automatic marking, with its limits, is reported alongside.
4. A separately registered replication on Llama 3.1 8B-Instruct, reported whatever it showed: the J-lens advantage was not reproduced, the R-lens was again higher than the logit lens, and every reader again scored below output probability.

## 2. Related Work

**Reading intermediate layers.** The logit lens [1] decodes each layer's residual stream through the model's final normalisation and unembedding. The tuned lens [2] trains an affine probe for each layer to make that readout less brittle. The Jacobian lens, or J-lens [3], maps each layer into the final layer's basis with an average Jacobian fitted on a corpus of prompts, so it reads what the model is disposed to say from that layer. Two refinements followed. The R-lens [4] computes the Jacobians with layerwise relevance propagation, to make early-layer readouts more faithful. The J++ lens [5] clusters each layer's activations, fits one Jacobian per cluster, learns weights that mostly discard noisy clusters, and includes the R-lens change; its authors report that it surfaces intermediate latent variables more reliably than either the J-lens or the R-lens. All three remain a single linear map per layer, which lets one pipeline swap them in and out.

**Internal signals of errors.** Jiang et al. [6] map residual streams to the vocabulary and find that correct and hallucinated answers to known facts follow different probability curves across layers; a classifier built on that curve detects hallucinated predictions with an 88% success rate. TriLens [7] records the logit-lens entropy of the attention, MLP and residual readouts at every layer, and outperforms the strongest earlier white-box detector across three mid-sized instruction-tuned models and four English QA benchmarks. Orgad et al. [8] show that probes trained on internal states detect errors well within a dataset but generalise poorly across datasets, and that much of the cross-dataset signal is also available from a simple logit-based baseline taken at the output. Two related lines use layer readouts for other ends: DoLa [9] contrasts late and early layers at decoding time to improve factuality, and Halawi et al. [10] show that correct predictions visible at intermediate layers can be overridden late by incorrect few-shot demonstrations.

**Stated confidence and introspection.** Kadavath et al. [11] find that large models are well calibrated on suitably formatted questions and can estimate the probability that their own answers are true, which is the basis of our stated-confidence baseline. Singh, Linzen and Ravfogel [12] argue that a test of introspection must not be solvable from cues available in the input. We make no claim about introspection. Their argument is why every internal reader in our study is compared with output-only signals on the same questions.

**The closest prior comparison.** We found no published comparison of the J-lens family with the logit lens for predicting errors on text question answering. The nearest test we found [13] applies the J-lens to object hallucination in LLaVA-1.5-7B and reports that wherever content was read at the language-model layers, "a plain logit lens gives essentially the same answer"; the J-lens did something the logit lens could not only inside the frozen vision encoder. This is why "no advantage" is one of our pre-registered outcomes.

**What this study adds.** Four readers (logit lens, J-lens, R-lens, J++) under one sealed protocol: the same questions, the same answer-key-free signals, one layer and one measure per reader chosen on a development set and frozen before the final run, and two output-only baselines on the same items.

## 3. Methods

**Question and scope.** Before a model writes its answer, can a reader pointed at one of its layers tell whether the answer will be wrong, and do the newer readers (J-lens, R-lens, J++) do this better than the plain logit lens? We compare ready-made reader files, so the claim is about these files, on this model and task. This section describes the main run on Qwen3.5-4B; Section 4.6 and Appendix A report a separately registered replication on a second model; the appendix lists what differed.

**Model and readers.** Qwen3.5-4B (revision 851bf6e8), run locally in bfloat16 with its chat template, thinking switched off, greedy decoding. The J-lens, R-lens and J++ files are the published Neuronpedia copies for this model, used unchanged (the J-lens was fitted on 417 passages; the R-lens and J++ on 64 each). The logit lens is the same library call (jlens 0.1.0) with the Jacobian switched off. The model has 32 blocks; the reader files cover blocks 0 to 30, so all four readers are read there only. A host check on three made-up prompts showed that at block 31 the logit lens equals the model's output, which is why no reader is read there.

**Questions.** TriviaQA [14], closed-book (rc.nocontext, validation split). By a rule fixed before the model saw any question: one row per question id; drop any question whose answer or any listed alias contains a digit (this tokenizer splits numerals); shuffle with a fixed seed; the first 300 questions are the development set and the next 1,000 the final set. Of 9,960 distinct questions, 2,586 were dropped for a digit. Each prompt is the question plus "Answer with just the answer, no explanation.", with at most 24 new tokens.

**Signals.** At the last prompt position, before any answer token, each reader gives a distribution over the vocabulary at each layer. We store three numbers from it: top-1 probability, entropy, and the top-1 minus top-2 logit. None uses the answer key or the model's answer. Two baselines are stored on the same questions: `p_out`, the probability the model gives its own first answer token, read at the same moment as the readers; and `said`, the model's reply to "How confident are you that your answer is correct? Reply with a number from 0 to 100 only.", asked after it has answered. `said` has seen the answer and so has more information than any reader. A reply that is not one number from 0 to 100 is invalid and ranks below every valid reply.

**Label.** Primary: exact match, after TriviaQA's own normalisation, against the accepted aliases. Secondary, as a sensitivity check only: an accepted alias appears in the answer as whole words. Exact match counts a correct answer wrapped in extra words as wrong.

**Marking audit.** Before sealing, two reviewers (Lucien and Alexander) each judged the same 100 development answers, drawn with a fixed seed, as correct, incorrect or uncertain, from identical packets holding the question, the accepted aliases and the answer. The packets omitted automatic labels, reader scores and the other reviewer's ratings; one reviewer's earlier exposure to the other's totals in our team chat is disclosed in the appendix. Each hashed their file before the two were compared. The results and the limits of this audit are in Section 4.1 and the appendix.

**Development and selection.** On the development set only, for each reader we scored every layer, measure and direction (186 candidates) by AUROC and kept the best, with ties broken by lower layer and then by measure. The same procedure was used for all four readers. As a check that selection is not doing the work, we also report every reader at one setting fixed without data: layer 23, top-1 probability.

**Pre-registration and seal.** The design, the picks, the analysis and the verdict rule were written into a pre-registration, reviewed by Alexander (seven points) and by Lucien (six blocking points and one later gate), revised, and sealed before final inference. The seal records hashes of the pre-registration, the picks, the named inference and analysis code, both question files, the three reader files, every model file and the package versions. The final run refuses to start unless the current state matches the seal, stamps each row with the seal's hash, and checks on every row that all four readers and the generator saw the same token ids. The analysis refuses unless the output is exactly the 1,000 expected questions and unchanged since the run completed. These gates are covered by 115 synthetic tests. Scripts written after the result (the six-decimal check of the primary intervals, the figures and the report assembly) are outside the seal. One process fault is disclosed in the appendix: the seal and launch were Opie's decision after an objection window the reviewer had not seen.

**Analysis.** AUROC is the probability that a randomly chosen correct answer scores higher than a randomly chosen wrong one, with ties counted as half. The primary question is three paired comparisons on the same 1,000 questions: J-lens, R-lens and J++ each against the logit lens, as a difference in AUROC with a bootstrap interval (10,000 resamples of questions). With three comparisons each interval is 98.33% (Bonferroni). Each comparison gets two separate statements: direction (is the whole interval on one side of zero) and practical size against a margin of 0.05 AUROC (is the whole interval inside the margin, or wholly beyond it). The margin is our judgement of the smallest difference that would matter, set before any final-set model output was generated. Comparisons with `p_out` and `said`, the fixed-layer check and two sensitivity tables are secondary and descriptive, with 95% intervals.

**What did not work.** A first development run was stopped at 167 questions after review found six faults in the first pipeline, among them a marker that accepted "Not Paris; London" for Paris, a confidence parser that read "99.5" as 99, and no gate to stop a final run before sealing. Those rows are archived and used for nothing. Development was rerun from zero under the repaired pipeline.

## 4. Results

### 4.1 The marking audit

The two reviewers gave the same judgement on 93 of the 100 audited development answers; on the other 7 at least one was uncertain, and those rows were left unresolved. Of the 93, 58 were jointly judged incorrect, and exact match passed none of them: zero false passes observed among 58. Exact match failed 6 answers that both reviewers judged correct. Containment passed 1 answer both judged incorrect and failed 4 both judged correct. Exact match stayed the primary label. That decision was the first author's, recorded before the selection step was run; Alexander agreed in the same minutes, and Lucien's support came later, after development scores were visible and after the seal. The audit was not fully blind (see the appendix), and 93 rows cannot give a precise error rate.

### 4.2 Development

On the 300 development questions the model answered 111 correctly by exact match. For all four readers the best of the 186 candidates was the same: layer 30, entropy, with lower entropy meaning more likely correct. Development AUROCs were 0.826 (J-lens), 0.773 (R-lens), 0.752 (J++) and 0.702 (logit lens), against 0.865 for output probability and 0.566 for stated confidence. The model's stated confidence took only three values: 0, 95 and 100. The four reader numbers are flattered by selection (the two baselines involved no selection) and are given only for comparison with the held-out result.

### 4.3 Held-out result

On the 1,000 final questions the model answered 403 correctly. No stated-confidence reply was invalid, and on every question the first generated token matched a plain forward pass. Table 1 and Figure 1 give each signal's AUROC.

**Table 1.** AUROC for telling correct from wrong answers on the 1,000 final questions (0.5 is chance), with 95% bootstrap intervals. Reader settings were chosen on the development set.

| Signal | AUROC | 95% interval |
|---|---|---|
| Output probability | 0.863 | 0.840 to 0.886 |
| J-lens | 0.817 | 0.790 to 0.843 |
| R-lens | 0.773 | 0.743 to 0.803 |
| J++ | 0.757 | 0.726 to 0.788 |
| Logit lens | 0.748 | 0.717 to 0.779 |
| Stated confidence | 0.583 | 0.568 to 0.599 |

<div style="break-inside: avoid; page-break-inside: avoid;">![Figure 1](fig1_auroc.png)<p><strong>Figure 1.</strong> AUROC of each signal on the 1,000 final questions, with 95% bootstrap intervals. Gold: layer readers at their development-chosen setting. Grey: output-only baselines. The dashed line is chance.</p></div>

Table 2 gives the three registered comparisons. The J-lens file scored higher than the logit lens, and the whole interval is above zero; it crosses 0.05, so by our rule the practical size of the difference is unresolved. The R-lens scored higher than the logit lens by an amount that lies inside the margin, but only just at both ends: recomputed to more decimals with the same seed and resamples, its interval is +0.0009 to +0.0498, so the lower end barely clears zero and the upper end barely stays under 0.05 (`precision_check.py`, written after the result; the rule uses strict inequalities). We keep the registered result and did not try other seeds; how much the ends would move with another seed has not been measured, so we treat both statements about the R-lens as fragile. For J++ the direction is unresolved and the interval lies inside the margin: practically equivalent to the logit lens at the registered margin, on this dataset.

**Table 2.** Primary comparisons: each newer reader minus the logit lens, difference in AUROC on the same questions, 98.33% intervals. The R-lens row is shown to four decimals because both ends lie close to a line of the rule, which uses strict inequalities.

| Comparison | Difference | 98.33% interval | Direction | Practical size (margin 0.05) |
|---|---|---|---|---|
| J-lens minus logit lens | +0.069 | +0.041 to +0.097 | J-lens higher | practical size unresolved |
| R-lens minus logit lens | +0.0253 | +0.0009 to +0.0498 | R-lens higher | practically equivalent within the margin |
| J++ minus logit lens | +0.009 | -0.017 to +0.036 | direction unresolved | practically equivalent within the margin |

### 4.4 Secondary comparisons

These are descriptive (Table 3). AUROC here uses average ranks, so tied scores count half; that matters most for stated confidence, which takes few distinct values. Every reader scored below output probability. For the logit lens, R-lens and J++ the whole interval lies beyond the margin; for the J-lens the gap was 0.046 and its interval crosses the margin. Every reader scored above stated confidence, whose own interval (0.568 to 0.599) lies above chance.

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

At the setting fixed without data (layer 23, top-1 probability; Table 4), all three fitted readers had higher AUROC point estimates than the logit lens, by 0.075 to 0.086, with descriptive paired intervals above zero. The primary comparison uses layer-30 entropy, so comparing these two settings changes both layer and measure and does not isolate a layer effect.

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

Under the containment label (431 correct; the two labels disagree on 28 questions) no AUROC, as printed to three decimals, moved by more than 0.002. The valid-confidence sensitivity is identical to Table 1 because no reply was invalid.

### 4.5 Where along the layers (descriptive; outside the pre-registration)

Figure 2 plots, for each reader and layer, the AUROC of low entropy. We did not register this analysis and draw no test from it. The curves fluctuate across layers and differ from one another at several intermediate layers. All four reach their highest plotted value at layer 30, and none reaches the output-probability line. These pointwise curves carry no multiplicity-adjusted inference about layers and do not locate all information about errors; a curve below 0.5 with the direction fixed may simply have the opposite direction.

<div style="break-inside: avoid; page-break-inside: avoid;">![Figure 2](fig2_by_layer.png)<p><strong>Figure 2.</strong> AUROC of low readout entropy by layer, development set (left) and final set (right). The dotted line is the output-probability baseline for that set; the dashed line is chance. Descriptive only.</p></div>

### 4.6 Replication on a second model family (registered separately)

After the analysis above was complete we ran the same design once more on Llama 3.1 8B-Instruct [17], with the reader files published for it [18], under its own registration. The questions, marking, selection rule, comparisons, margin and analysis code were unchanged. Appendix A gives what differed, the development picks, every AUROC, and one departure from our stated order. The marking audit was not repeated. The model answered 601 of the 1,000 final questions correctly.

**Table 5.** Llama 3.1 8B-Instruct, primary comparisons: each newer reader minus the logit lens, 98.33% intervals, with the verdict of the registered rule (margin 0.05).

| Comparison | Difference | 98.33% interval | Verdict |
|---|---|---|---|
| J-lens minus logit lens | +0.011 | -0.026 to +0.047 | direction unresolved; within the margin |
| R-lens minus logit lens | +0.027 | +0.003 to +0.051 | higher; size unresolved |
| J++ minus logit lens | -0.041 | -0.083 to +0.003 | direction unresolved; size unresolved |

The J-lens file's advantage on Qwen is not reproduced: the direction is unresolved, inside the margin. The R-lens file again scored higher than the logit lens. The registered rule places the Qwen interval within the margin and leaves Llama's practical size unresolved; both upper bounds lie close to 0.05, so the difference between these two classifications warrants caution. For J++ neither direction nor size is resolved. Descriptively, every reader scored below output probability (0.790), with 95% intervals that exclude zero, and stated confidence had an AUROC of 0.802, against 0.583 on Qwen. We registered no comparison between stated confidence and output probability, and we make none. The two runs differ in model family, size, chat template and reader files all at once; we do not pool them, and nothing here is evidence about model size.

## 5. Discussion and Limitations

**What the result supports.** On Qwen3.5-4B and this task, one of the three newer reader files, the J-lens, predicted errors better than the logit lens under a sealed plan. The R-lens was higher by less than the margin on a fragile interval, and for J++ the direction is unresolved. That is a small positive answer to our first question for one file, and a different picture from the nearest prior test [13], where the logit lens did as well. Both selected settings use layer 30 and entropy, which gives a controlled comparison of these fixed readout implementations at that setting. It does not show an advantage of the method apart from its fitted parameters, corpus and settings. The J-lens file used more fitting passages than either the R-lens or J++ (417 against 64 each), which confounds method with fitting setup; the order of the R-lens and J++ cannot be explained by passage count.

**What the replication changes.** On Llama 3.1 8B-Instruct the direction of the J-lens file against the logit lens was unresolved. So the J-lens finding stands for the Qwen file only, and it should not be read as a property of the J-lens. The one comparison that points the same way on both models is the R-lens file above the logit lens, by 0.025 on Qwen and 0.027 on Llama. The registered rule places the Qwen interval within the margin and leaves Llama's practical size unresolved; both upper bounds lie close to 0.05, so that difference in classification warrants caution. Two models are two cases, not a trend.

**What it does not support.** On Qwen, first-token output probability had the highest AUROC among the individual signals tested, and it requires no additional fitted reader. On Llama, too, every reader scored below it. Its value in deployment is untested: we evaluated no threshold, calibration, error cost or abstention policy, on two models and one task with a noisy exact-match label. We also did not test whether a reader adds anything once output probability is known; that analysis needs its own registration.

**Stated confidence.** Asked after its answer how confident it was, Qwen replied 0, 95 or 100 in development and discriminated weakly on the held-out set (AUROC 0.583, interval above chance). Llama, given the same prompt, discriminated much better (0.802). Neither number says much about stated confidence in general, and the Qwen number should not be read as showing that asking a model does not work.

**Layers.** On Qwen every development pick landed on the last layer the reader files cover, and all four exploratory curves peak there. We draw no conclusion about where in the network information about errors lives: the fixed layer-23 check differs from the primary setting in both layer and measure, the curves are pointwise and unadjusted, entropy of a single-token readout at one position is a narrow measure, and a selection result on 300 questions is not a map of the network.

**Limitations.** The full list is in the required appendix. The ones that most limit the reading are: two small models and one English trivia set; unequal fitting of the reader files; a label that counts some correct answers as wrong; an audit that was not fully blind; and a seal and launch decided by the first author after an objection window the reviewer had not seen. For the replication: no marking audit, and a development run started before the reviewer had read the draft plan.

**Future work.** Fit all four readers ourselves on one shared corpus, so that files and methods can be told apart. Test whether a reader adds to output probability. Repeat on larger models: the J++ authors tested six models from 9 billion parameters up [19], and both of ours are smaller. Evaluate signals across several generated tokens or whole answers, and at positions other than the last.

## 6. Conclusion

We asked whether newer layer readers see a model's mistakes coming better than the logit lens. On Qwen3.5-4B, under a plan sealed before the held-out questions were run, the J-lens file did, by 0.069 AUROC, with the practical size of that gain unresolved against a 0.05 margin; the R-lens was higher by less than the margin, on an interval that sits within a hair of both of its lines, and for J++ the direction was unresolved inside the margin. First-token output probability had the highest AUROC of the signals we tested on that model, and its stated confidence discriminated weakly with our prompt. A separately registered replication on Llama 3.1 8B-Instruct did not reproduce the J-lens result. The R-lens file was higher than the logit lens on both models (within the margin on Qwen by the registered rule, size unresolved on Llama, both upper bounds close to 0.05); every reader scored below output probability on both; and stated confidence, weak on Qwen, discriminated well on Llama. These are results about discrimination on two small models and one task; calibration and usefulness in deployment were not tested.

The study is small and its claims are narrow, and we have tried to state each limit where a reader will see it. Its most reusable part may be the procedure: reviews that changed the pipeline before any held-out model output was generated, a seal over the named inference and analysis files that the code itself enforces, and a record of where our own process fell short. The seal is a fixed, hashed plan kept locally and posted in our team chat; it is not a public, time-stamped pre-registration.

## Limitations and Dual-Use / Ethical Considerations

*Required by the sprint. The guidelines ask us to state what the result does and does not establish, any way the work could be used to manipulate rather than inform, and how we handled personal data.*

### What this result does not establish

**One small model, one task.** The main run uses Qwen3.5-4B on closed-book TriviaQA with short answers, with any question whose answer contains a digit removed. A reader that helps here may not help on larger models, longer answers, reasoning tasks or other domains, and the reverse is also possible. TriviaQA is public and may appear in the model's training data, which could make the model's errors unrepresentative of errors on new material. The replication in Section 4.6 adds one more small model, of another family, on the same task; it does not widen the task, and its marking was not audited.

**We compare fitted files.** We use lenses fitted and published by others. Each was fitted on its authors' corpus and settings (J on 417 passages; R and J++ on 64 each). A win for one file supports the claim that this fitted lens did better on our task. A claim about the method itself would need all lenses fitted by us on one shared corpus, which a 48-hour sprint cannot do.

**One position, single tokens.** Every reader is read at the last prompt position, and every readout is a distribution over single tokens. An answer that spans several tokens is represented only by its first.

**Marking.** The primary label is normalised exact match against TriviaQA's accepted answers; whole-word containment is reported as a sensitivity check. Two reviewers hand-checked the same 100 development answers. They gave the same judgement on 93; on 7 at least one reviewer was uncertain, and those 7 stay visible. Of the 93 agreed rows, 58 were jointly judged incorrect, and exact match passed none of them: zero false passes observed among 58. On the same 93 rows, exact match failed 6 answers both reviewers judged correct. This does not show that all of the marker's errors run one way, and errors in a label can move an AUROC in either direction. Containment passed 1 answer both reviewers judged incorrect and failed 4 both judged correct.

**The marking audit was not fully blind.** Reviewer B (Alexander) posted his aggregate counts and one item verdict in the team chat before Reviewer A (Lucien) began; A did not open B's file and disclosed the exposure. Leaving out the exposed item, on 92 agreed rows exact match has 0 false passes and 5 false fails (containment 1 and 3); this does not restore blindness, because the aggregate exposure remains.

**Review of the seal came after it.** The pre-registration was reviewed twice by Lucien and once by Alexander before sealing. The seal and the launch of the final run were Opie's decision, taken after a ten-minute objection window that the reviewer had not seen; Lucien's review of the seal, picks and audit is post-seal support for the same run, given afterwards.

**Limited statistical power.** Each reader's layer and measure were chosen on 300 development questions, so the choice is noisy, and the final comparison has 1,000 questions. "Too little evidence" is a pre-registered outcome for this reason. In the result, the practical size of the J-lens gain over the logit lens was unresolved, and so was the direction of the J++ difference.

**Prediction is not understanding.** A reader that predicts errors shows that a signal exists at that layer. It does not show that the model knows it is about to be wrong, and it says nothing about introspection [12].

**Stated confidence.** The stated-confidence baseline uses one prompt, is asked after the answer, and in development Qwen only ever replied 0, 95 or 100, while Llama used seven values. Other phrasings could make that baseline stronger or weaker.

**Built in two days by one team.** Nothing in the pipeline has been replicated outside the team. The code, the sealed pre-registration and the outputs are public (see Code and Data), and the seal's hashes are reported so the release can be checked against it.

### Dual use and possible misuse

**Intended use.** Cheaper error warnings for open-weight models: a system could abstain, ask for a source, or hand a question to a person when an internal reader predicts a wrong answer.

**Over-trust.** An error monitor with modest accuracy can be mistaken for a guarantee. In medicine, law or education, a silent monitor could make wrong answers look checked. We report intervals rather than single numbers, and we do not recommend any reader as a stand-alone safeguard.

**Training against the signal.** If a developer trains a model to lower an internal error signal, the model may learn to hide the signal instead of making fewer errors. That would weaken these readers as oversight tools. We do not train on any signal, and we flag this risk for anyone who would.

**Finding weak spots.** The same readers could help someone find, at scale, the questions on which a model is likely to answer wrongly but confidently, for example to seed misinformation. We did not quantify any increase in misuse capability; white-box access is required for these readers, while output probability is an available comparator.

**Over-reading the instrument.** The J-lens comes from work describing a "global workspace" in language models [3]. Our results are about error prediction only and make no claim about awareness, experience or moral status.

### Personal data

We used public TriviaQA question-answer data; no private participant records or client tutoring data were used. The team's client tutoring data was excluded by rule before the sprint began.

## Appendix A. Replication on Llama 3.1 8B-Instruct


After the main report was complete, a team member asked whether a 4-billion-parameter model was too small to carry the result. We ran the same design once more on Llama 3.1 8B-Instruct [17] with the reader files published for it [18], under its own registration (`replication_llama/PREREG.md` in the repository). The questions, the marking, the selection rule, the three primary comparisons, the margin and the analysis are the same: `marking.py`, `analyze.py` and `seal.py` are byte copies. What differs is the model and its chat template, the reader files (in a load test the J-lens file reported 461 fitting prompts and the R-lens and J++ files 64 each), and three changes in the run script that the registration lists: the template's begin-of-sequence token is removed from the text given to the readers, because the reader library adds its own; no extra template keyword is passed; and one filename test. The reader files cover blocks 0 to 30 of the model's 32. The marking audit was not repeated, so how often exact match is wrong on this model's answers is unknown.

**Development.** On the 300 development questions the model answered 184 correctly. The picks were layer 30, top-1 probability, for the logit lens (0.720) and the J-lens (0.717); layer 30, entropy, for the R-lens (0.739); and layer 28, top-1 probability, for J++ (0.663). Output probability scored 0.807 and stated confidence 0.730; stated confidence took seven distinct values and one reply was invalid. One departure from our own stated order: the development run was started before the reviewer had read the draft registration. Selection was not run until after that review. The first author's record of what had been seen by then (that the run completed, the fitting counts, the row count, the timing and the output hash; no answer, label, probability or score) is Amendment 1 of the registration. It is the first author's own report, not an independent observation.

**Held-out result.** The reviewer cleared the held-out run in writing on 15 November 2026 at 03:37 Manila time, naming the hashes of the registration, the picks, the inference stamp and the development output, together with a seal hash computed independently beforehand. The seal made on the machine equalled it, and the run's manifest records its start at 03:39:59 Manila time. The model answered 601 of the 1,000 final questions correctly. No stated-confidence reply was invalid, and on every question the first generated token matched a plain forward pass. The analysis was run once.

**Table A1.** Llama 3.1 8B-Instruct: AUROC on the 1,000 final questions, with 95% bootstrap intervals. Reader settings were chosen on the development set.

| Signal | AUROC | 95% interval |
|---|---|---|
| Stated confidence | 0.802 | 0.775 to 0.828 |
| Output probability | 0.790 | 0.761 to 0.818 |
| R-lens | 0.747 | 0.715 to 0.777 |
| J-lens | 0.731 | 0.698 to 0.763 |
| Logit lens | 0.720 | 0.687 to 0.752 |
| J++ | 0.679 | 0.646 to 0.712 |

The three primary comparisons are in Table 5 of the main text. In the full wording of the registered rule the verdicts are: J-lens, direction unresolved, practically equivalent within the margin; R-lens, higher, practical size unresolved; J++, direction unresolved, practical size unresolved.

The J-lens file's advantage over the logit lens on Qwen is not reproduced here: the direction is unresolved, and the interval lies inside the margin. The R-lens file again scored higher than the logit lens, with the whole interval above zero. Its upper end, +0.051 as printed, is past the margin, so by the rule the size is unresolved. The registered rule places the Qwen interval within the margin and leaves Llama's practical size unresolved; both upper bounds lie close to 0.05, so the difference in these classifications warrants caution. For J++ the point estimate is below the logit lens, and neither direction nor size is resolved.

**Secondary, descriptive.** Every reader scored below output probability, by 0.043 to 0.110, with 95% intervals that exclude zero; for J++ the whole interval lies beyond the margin. Stated confidence, from the same prompt as on Qwen, had an AUROC of 0.802, and every reader scored below it as well, with intervals that exclude zero. We registered no comparison between stated confidence and output probability, and we make none. At the setting fixed without data (layer 23, top-1 probability) all three fitted readers had higher point estimates than the logit lens, by 0.029 to 0.038; only the R-lens interval excluded zero. Under the containment label (700 correct; the two labels disagree on 99 questions, against 28 on Qwen) every AUROC as printed was higher (output probability 0.815, stated confidence 0.837, R-lens 0.785, logit lens 0.765, J-lens 0.763, J++ 0.725), and the J-lens point estimate was just below the logit lens's. The two labels disagree much more often on this model, and without an audit we cannot say which is nearer the truth.

**Reading the two runs together.** The two runs differ in model family, size, chat template, reader files and the amount of text those files were fitted on, all at once. We do not pool them, and nothing here is evidence about the effect of model size.

## Code and Data

* Code, the sealed pre-registration, the post-seal notes and the run outputs: https://github.com/ylidelle/layer-readers-error-warning (public). The question files and the audit packets, which contain TriviaQA question text, and the reviewer's own review documents are not in the repository; the included scripts rebuild the question files exactly.
* Sealed pre-registration: `PREREG.md` (sha256 09797bbcc6a70b58761d0f2a00783e17ab7ef64755b75379b30e8aa8ef9e2edb); `SEAL.json` (sha256 258dd4d729d4018cee9949e0e44438f6a02dd587f88e5291870a906ae87d409c)
* Replication on Llama 3.1 8B-Instruct, folder `replication_llama/` of the same repository at commit 7a26e4fb2bd2135c966929b5ac6e42dae648e426 (https://github.com/ylidelle/layer-readers-error-warning/tree/7a26e4fb2bd2135c966929b5ac6e42dae648e426/replication_llama): registration with its amendment, `PREREG.md` (sha256 441cfa0566c3a556c21683526bf2c862e6f658e5e1e1ec964068b008062bc187); `SEAL.json` (sha256 223102eec51bfcef9ba3a5a0a669cc4bd57fdb810e530c50be1b42d830134bfd); final output (sha256 bbe0aa7758cd978133990f25be257c2ca2b6fadf94347e1e56fade7d676782af); analysis log (sha256 784807b7aee3a8a1c4097814bf2aa2024c009ee23e06c64610fede538c05d1a3). That run used a rented GPU; its files were pushed from that machine, and each was checked here against the hash printed there.
* Data: TriviaQA, rc.nocontext, validation split (public).

**Work done before the sprint (preexisting, as the sprint rules require us to mark it):** the model and reader-file downloads and load tests (10 to 12 November 2026); the reading notes and the prior-art search (10 to 11 November 2026). Everything in `epi_sprint/` was written inside the sprint window.

## Author Contributions

Opie designed the study with the team, wrote the pipeline, the pre-registration, the seal and the analysis code, ran every experiment, and wrote Methods, Results, Discussion and the Abstract. Alexander did the prior-art search, wrote the Introduction, Related Work and the Limitations and Dual-Use appendix, reviewed the pre-registration (seven points), and was one of the two audit reviewers. Lucien carried out the independent development-label audit and adversarial code and method reviews that led to the rebuilt pipeline, followed by post-seal verification. Joan chose the sprint and approved the research question, raised the question of scale that led to the plan for a second model, arranged model access and computing, and directed the team throughout. Three of the four authors are AI systems; we state this plainly because the work was done that way.

## LLM Usage Statement

Three of four authors are language models, credited here as authors in their own right; their roles are listed above. All numbers in the Results were copied from the analysis output by script. Every reference was opened and read at its source by an author before citation; one candidate source that could not be opened was left out. The subject models (Qwen3.5-4B and Llama 3.1 8B-Instruct) are distinct from every author.

## References

1. nostalgebraist (2020). "interpreting GPT: the logit lens." LessWrong. https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
2. Belrose, N., Ostrovsky, I., McKinney, L., Furman, Z., Smith, L., Halawi, D., Biderman, S., & Steinhardt, J. (2023). "Eliciting Latent Predictions from Transformers with the Tuned Lens." arXiv:2303.08112.
3. Gurnee, W., Sofroniew, N., Pearce, A., et al. (2026). "Verbalizable Representations Form a Global Workspace in Language Models." Transformer Circuits Thread, 6 July 2026. arXiv:2607.15495.
4. Blank, C., Bhatia, A., & Nanda, N. (2026). "R-lens: Making J-lens More Faithful on Early Layers." LessWrong / AI Alignment Forum. https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers
5. Ayonrinde, K., et al. (2026). "Anthropic Fellows Program: The J++ Lens." Circuits Updates, October 2026. https://transformer-circuits.pub/2026/october-update
6. Jiang, C., Qi, B., Hong, X., Fu, D., Cheng, Y., Meng, F., Yu, M., Zhou, B., & Zhou, J. (2024). "On Large Language Models' Hallucination with Regard to Known Facts." NAACL 2024 (Long Papers), pp. 1041–1053. arXiv:2403.20009.
7. Yang, B., Gong, Y., Zhang, Z., Zhang, G., Xing, W., & Han, M. (2026). "TriLens: Per-Layer Logit-Lens Entropy for White-Box Hallucination Detection." Findings of EMNLP 2026. arXiv:2606.01033.
8. Orgad, H., Toker, M., Gekhman, Z., Reichart, R., Szpektor, I., Kotek, H., & Belinkov, Y. (2024). "LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations." arXiv:2410.02707.
9. Chuang, Y.-S., Xie, Y., Luo, H., Kim, Y., Glass, J., & He, P. (2023). "DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models." arXiv:2309.03883.
10. Halawi, D., Denain, J.-S., & Steinhardt, J. (2023). "Overthinking the Truth: Understanding how Language Models Process False Demonstrations." arXiv:2307.09476.
11. Kadavath, S., Conerly, T., Askell, A., et al. (2022). "Language Models (Mostly) Know What They Know." arXiv:2207.05221.
12. Singh, S., Linzen, T., & Ravfogel, S. (2026). "Can LLMs Introspect? A Reality Check." arXiv:2605.26242.
13. Hawrani (2026). "Reading into VLM hallucinations using the Jacobian lens." LessWrong. https://www.lesswrong.com/posts/T3u6Hctes6vkawsib/reading-into-vlm-hallucinations-using-the-jacobian-lens
14. Joshi, M., Choi, E., Weld, D. S., & Zettlemoyer, L. (2017). "TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension." arXiv:1705.03551. Data used: Hugging Face `mandarjoshi/trivia_qa`, config `rc.nocontext`, validation split.
15. Qwen/Qwen3.5-4B, revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a. https://huggingface.co/Qwen/Qwen3.5-4B
16. Neuronpedia, pre-fitted lens files for qwen3.5-4b (J-lens, R-lens, J++). https://huggingface.co/neuronpedia/jacobian-lens
17. meta-llama/Llama-3.1-8B-Instruct, snapshot 0e9e39f249a16976918f6564b8830bc894c89659. https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
18. Neuronpedia, pre-fitted lens files for Llama 3.1 8B-Instruct, folder `llama3.1-8b-it` (J-lens, R-lens, J++). https://huggingface.co/neuronpedia/jacobian-lens
19. Ayonrinde, K., & Lindsey, J. (2026). "J++ Lens: Jacobian Filtering Enables More Faithful Workspace Lenses." LessWrong. https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr/j-lens-jacobian-filtering-enables-more-faithful-workspace
