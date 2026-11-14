"""SPRINT marking audit (2026-11-14; Lucien's review, point 3). DEVELOPMENT rows only.
  make_audit.py sheet     write two identical blank sheets of 100 seeded development rows: question, accepted aliases,
                          the model's answer.  NO automatic label, NO reader score, NO probability.
  make_audit.py compare   after both reviewers have filled "judgement" (correct / incorrect / uncertain, with a reason), working apart:
                          agreement between them, and each label rule against their agreed judgements.
The sample: random.Random(AUDIT_SEED).sample(sorted development ids, AUDIT_N)."""
import sys, json, random
import config, seal, analyze
sys.stdout.reconfigure(encoding="utf-8")
B = config.BASE
def sample_ids():
    return random.Random(config.AUDIT_SEED).sample(sorted(seal.expected_qids("dev")), config.AUDIT_N)
if sys.argv[1] == "sheet":
    import os
    if os.path.exists(B + "out/audit_packet_ORIGINAL.json"): raise seal.Refused("an audit packet already exists; it is made once")
    vrows, vstamp, vsha = analyze.verified_dev()
    rows = {r["qid"]: r for r in vrows}
    qs = {json.loads(l)["qid"]: json.loads(l) for l in open(B + "data/dev.jsonl", encoding="utf-8")}
    sheet = [{"qid": q, "question": qs[q]["question"], "accepted_aliases": qs[q]["aliases"], "model_answer": rows[q]["answer"],
              "judgement": "", "reason": ""} for q in sample_ids()]
    packet = {"what": "development rows only; no automatic label, no confidence, no reader score, no picks",
              "sampling_rule": f"random.Random({config.AUDIT_SEED}).sample(sorted(development question ids), {config.AUDIT_N})",
              "repaired_run_inference_stamp": vstamp, "dev_output_sha256": vsha, "rows_sha256": seal.sha_obj(sheet),
              "rubric": "judgement = correct / incorrect / uncertain, with a short reason (reviewers' own agreed rubric, Kwentuhan 2026-11-14)",
              "rows": sheet}
    print("packet content sha256 (rows):", seal.sha_obj(sheet))
    for who in ("ORIGINAL", "reviewer_A", "reviewer_B"):
        json.dump(packet, open(B + f"out/audit_packet_{who}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"blank packets written ({len(sheet)} rows): ORIGINAL (never edited), reviewer_A, reviewer_B | file sha256 {seal.sha_file(B + 'out/audit_packet_ORIGINAL.json')}")
elif sys.argv[1] == "compare":
    vrows, vstamp, vsha = analyze.verified_dev()
    orig = json.load(open(B + "out/audit_packet_ORIGINAL.json", encoding="utf-8"))
    if orig["dev_output_sha256"] != vsha or orig["rows_sha256"] != seal.sha_obj(orig["rows"]) or [r["qid"] for r in orig["rows"]] != sample_ids():
        raise seal.Refused("the original packet is not the registered sample of the verified development output")
    keep = ("qid", "question", "accepted_aliases", "model_answer")
    files = {w: B + f"out/audit_packet_reviewer_{w}.json" for w in ("A", "B")}
    A, Bv = (json.load(open(files[w], encoding="utf-8"))["rows"] for w in ("A", "B"))
    for w, R in (("A", A), ("B", Bv)):
        if [{k: r[k] for k in keep} for r in R] != [{k: r[k] for k in keep} for r in orig["rows"]]:
            raise seal.Refused(f"reviewer {w}'s file does not hold the original packet content unchanged")
        print(f"reviewer {w} ratings file sha256 {seal.sha_file(files[w])}  (each reviewer confirms this is the hash they recorded)")
    ok = {"correct", "incorrect", "uncertain"}
    if any(r["judgement"] not in ok for r in A + Bv): raise seal.Refused("every row needs a judgement: correct, incorrect or uncertain")
    rows = {r["qid"]: r for r in vrows}
    n = len(A)
    same = sum(a["judgement"] == b["judgement"] for a, b in zip(A, Bv))
    unc = sum("uncertain" in (a["judgement"], b["judgement"]) for a, b in zip(A, Bv))
    agree = [(a["qid"], a["judgement"]) for a, b in zip(A, Bv) if a["judgement"] == b["judgement"] and a["judgement"] != "uncertain"]
    print(f"ALL {n} ROWS: same judgement on {same}; at least one 'uncertain' on {unc}; different judgements on {n - same}")
    print(f"DETERMINATE-AGREEMENT SUBSET: {len(agree)} of {n} rows where both reviewers gave the same sure judgement. "
          f"Marker errors below use THIS denominator; it is not an unbiased estimate for all {n}.")
    for rule in ("correct_em", "correct_contain"):
        fp = [q for q, j in agree if rows[q][rule] and j == "incorrect"]; fn = [q for q, j in agree if not rows[q][rule] and j == "correct"]
        print(f"{rule}: marked correct but judged incorrect: {len(fp)} of {len(agree)}; marked wrong but judged correct: {len(fn)} of {len(agree)}")
        print("   ids marked correct, judged incorrect:", fp); print("   ids marked wrong, judged correct:", fn)
    print("rows without determinate agreement:", [(a["qid"], a["judgement"], b["judgement"]) for a, b in zip(A, Bv) if not (a["judgement"] == b["judgement"] != "uncertain")])
