"""SPRINT (written 2026-11-14, inside the sprint window). Build the development and final question sets.
Source: TriviaQA (mandarjoshi/trivia_qa, config rc.nocontext, split validation, parquet file 0),
local copy sha256 printed below.  FIXED RULE, written before any model saw any question:
  1. keep one row per question_id (first occurrence);
  2. DROP a question if its answer value or ANY listed alias contains a digit (scope limit: Qwen splits numerals);
  3. shuffle the kept rows with random.Random(20261114);
  4. first 300 = development set, next 1000 = final set.
The script prints counts and file hashes only. It prints NO final-set question."""
import sys, json, re, random, hashlib
import pyarrow.parquet as pq
sys.stdout.reconfigure(encoding="utf-8")
SRC = "E:/lab-data/datasets/trivia_qa/rc.nocontext_validation_0.parquet"
OUT = "E:/OneDrive/Claude Orion Bennett/Lab/epi_sprint/data/"
SEED, N_DEV, N_FINAL = 20261114, 300, 1000
sha = lambda b: hashlib.sha256(b).hexdigest()
print("source sha256:", sha(open(SRC, "rb").read()))
rows = pq.read_table(SRC, columns=["question", "question_id", "answer"]).to_pylist()
print("rows in file:", len(rows))
seen, kept, n_dup, n_digit = set(), [], 0, 0
for r in rows:
    if r["question_id"] in seen:
        n_dup += 1; continue
    seen.add(r["question_id"])
    a = r["answer"]
    names = [a["value"]] + list(a["aliases"]) + list(a["normalized_aliases"])
    if any(re.search(r"\d", s) for s in names):
        n_digit += 1; continue
    kept.append({"qid": r["question_id"], "question": r["question"].strip(), "answer": a["value"],
                 "aliases": sorted(set(a["normalized_aliases"]) | {a["normalized_value"]})})
print(f"duplicates dropped: {n_dup} | dropped for a digit: {n_digit} | kept: {len(kept)}")
assert len(kept) >= N_DEV + N_FINAL
random.Random(SEED).shuffle(kept)
dev, final = kept[:N_DEV], kept[N_DEV:N_DEV + N_FINAL]
assert not ({r["qid"] for r in dev} & {r["qid"] for r in final})
for name, part in (("dev", dev), ("final", final)):
    blob = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part).encode("utf-8")
    open(OUT + name + ".jsonl", "wb").write(blob)
    print(f"{name}: {len(part)} questions, sha256 {sha(blob)}")
print("dev example (dev only):", json.dumps(dev[0], ensure_ascii=False)[:300])
