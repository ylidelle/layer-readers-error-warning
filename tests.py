"""SPRINT tests (2026-11-14). CPU only, synthetic data only: no real question is read, no model is loaded.
Run:  python tests.py     and     python -O tests.py   (the gates must still refuse with asserts switched off)."""
import sys, os, json, shutil, tempfile, math, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/"
sys.path.insert(0, HERE)
import config, seal, marking, analyze
sys.stdout.reconfigure(encoding="utf-8")
N = [0]
def ok(cond, what):
    N[0] += 1
    if not cond: print("FAIL:", what); sys.exit(1)
def refuses(fn, needle, what):
    try: fn()
    except seal.Refused as e:
        ok(needle in str(e), f"{what}: refused, but for another reason: {e}"); return
    ok(False, f"{what}: did NOT refuse")

# ---- marking (Lucien's cases) ----
ok(not marking.exact_match("Not Paris; London", ["paris"]), "EM rejects 'Not Paris; London'")
ok(marking.contains_alias("Not Paris; London", ["paris"]), "containment accepts it (why it is secondary)")
ok(not marking.exact_match("Paris or London", ["paris"]), "EM rejects alternatives")
ok(not marking.exact_match("New York", ["york"]), "EM rejects New York for York")
ok(marking.exact_match("**Paris.**", ["paris"]), "EM ignores punctuation")
ok(marking.exact_match("The Beatles", ["beatles"]), "EM ignores articles")
ok(marking.exact_match("ST. PETERSBURG", ["st petersburg"]), "EM ignores case")
ok(not marking.exact_match("", ["a"]), "empty answer is wrong")
ok(not marking.exact_match("The answer is Paris", ["paris"]), "EM rejects a wrapped answer (known cost)")
for text, want in [("99.5", 99.5), ("-5", None), ("150", None), ("I considered 2 options; confidence 90", None),
                   ("90%", 90.0), ("0.95", 0.95), ("100", 100.0), ("", None), (" 85 ", 85.0), ("85 percent", None),
                   ("1e2", None), ("100.0%", 100.0), ("100.5", None)]:
    ok(marking.parse_confidence(text) == want, f"parse_confidence({text!r}) should be {want}")

# ---- verdicts (Lucien's point 5) ----
V = analyze.verdict
ok(V(0.01, 0.08) == ("A higher", "practical size unresolved"), "[0.01,0.08]")
ok(V(0.06, 0.10) == ("A higher", "A better by more than the margin"), "[0.06,0.10]")
ok(V(0.01, 0.04) == ("A higher", "practically equivalent within the margin"), "[0.01,0.04]")
ok(V(-0.02, 0.03) == ("direction unresolved", "practically equivalent within the margin"), "[-0.02,0.03]")
ok(V(-0.10, 0.10) == ("direction unresolved", "practical size unresolved"), "[-0.10,0.10]")
ok(V(-0.09, -0.06) == ("B higher", "B better by more than the margin"), "[-0.09,-0.06]")
ok(V(0.0, 0.04)[0] == "direction unresolved", "lower end exactly 0 is unresolved")
ok(V(0.05, 0.09)[1] == "practical size unresolved", "lower end exactly on the margin is unresolved")

# ---- AUROC against a pairwise oracle, with ties ----
rng = np.random.default_rng(0)
for _ in range(50):
    n = int(rng.integers(4, 30)); s = rng.integers(0, 5, n).astype(float); y = rng.integers(0, 2, n).astype(bool)
    if y.all() or not y.any(): continue
    pairs = [(a > b) + 0.5 * (a == b) for a in s[y] for b in s[~y]]
    ok(abs(analyze.auroc(s, y) - sum(pairs) / len(pairs)) < 1e-12, "auroc equals the pairwise oracle")
ok(math.isnan(analyze.auroc([1, 2], [True, True])), "one class gives nan")

# ---- integrity gates on a throwaway copy with tiny fake files ----
T = tempfile.mkdtemp(prefix="epi_test_").replace("\\", "/") + "/"
try:
    for n in ("config.py", "seal.py", "run_readers.py", "marking.py", "analyze.py", "build_sets.py"): shutil.copy(HERE + n, T + n)
    os.makedirs(T + "data"); os.makedirs(T + "out"); os.makedirs(T + "model"); os.makedirs(T + "lens")
    fake = {"dev": ["d1", "d2", "d3"], "final": ["f1", "f2", "f3", "f4"]}
    for s, ids in fake.items():
        open(T + f"data/{s}.jsonl", "w", encoding="utf-8").write("".join(json.dumps({"qid": q, "question": "synthetic?", "answer": "x", "aliases": ["x"]}) + "\n" for q in ids))
    open(T + "model/weights.bin", "wb").write(b"fake weights"); 
    for k in ("J", "R", "J++"): open(T + f"lens/{k}.pt", "wb").write(k.encode())
    config.BASE, config.MODEL, config.LENS_DIR = T, T + "model", T + "lens/"
    config.LENSES = {"J": "J.pt", "R": "R.pt", "J++": "J++.pt"}
    config.DATA_SHA = {s: seal.sha_file(T + f"data/{s}.jsonl") for s in fake}
    config.N_ROWS = {"dev": 3, "final": 4}
    import run_readers
    def row(qid, stamp, em=True, bad=None):
        d = {"p1": 0.5, "ent": 1.0, "margin": 0.1, "top": 1}
        r = {"qid": qid, "stamp": stamp, "answer": "x", "correct_em": em, "correct_contain": em, "p_out": 0.5, "p_out_raw": 0.5,
             "first_id": 1, "fwd_top": 1, "said": 90.0, "said_raw": "90", "n_tok": 3, "ids_sha": "0",
             "read": {rd: {str(L): dict(d) for L in config.LAYERS} for rd in config.READERS}}
        if bad == "nan": r["read"]["J"]["5"]["ent"] = float("nan")
        return r
    def put(name, rows, tail=b""):
        open(seal.out_path(name), "wb").write("".join(json.dumps(r) + "\n" for r in rows).encode() + tail)

    refuses(lambda: run_readers.main(["final"]), "no SEAL.json", "final run before sealing")
    refuses(lambda: run_readers.main(["final", "--limit", "2"]), "--limit", "limit on the final set")
    refuses(lambda: run_readers.main(["test"]), "dev or final", "unknown set name")
    stamp = seal.inference_stamp()
    # development completeness and schema
    put("dev", [row("d1", stamp), row("d2", stamp, em=False)])
    refuses(lambda: seal.load_complete("dev", stamp), "ids differ", "incomplete development set")
    put("dev", [row("d1", stamp), row("d2", stamp, em=False), row("d2", stamp)])
    refuses(lambda: seal.load_complete("dev", stamp), "duplicate", "duplicate question id")
    refuses(lambda: seal.check_resume(seal.read_rows(seal.out_path("dev")), "dev", stamp), "duplicate", "resume over duplicates")
    put("dev", [row("d1", stamp), row("d2", stamp, em=False), row("zz", stamp)])
    refuses(lambda: seal.load_complete("dev", stamp), "ids differ", "unexpected question id")
    put("dev", [row("d1", stamp), row("d2", stamp), row("d3", stamp)])
    refuses(lambda: seal.load_complete("dev", stamp), "one outcome class", "one class only")
    put("dev", [row("d1", stamp), row("d2", stamp, em=False, bad="nan"), row("d3", stamp)])
    refuses(lambda: seal.load_complete("dev", stamp), "not finite", "non-finite signal")
    put("dev", [row("d1", "other"), row("d2", stamp, em=False), row("d3", stamp)])
    refuses(lambda: seal.load_complete("dev", stamp), "carries stamp", "row from another stamp")
    refuses(lambda: seal.check_resume(seal.read_rows(seal.out_path("dev")), "dev", stamp), "different stamp", "resume into another stamp's rows")
    put("dev", [row("d2", stamp)])
    refuses(lambda: seal.check_resume(seal.read_rows(seal.out_path("dev")), "dev", stamp), "not a prefix", "resume out of order")
    put("dev", [row("d1", stamp)], tail=b'{"qid": "d2", "sta')
    refuses(lambda: seal.read_rows(seal.out_path("dev")), "torn tail", "torn last line")
    ok(len(seal.read_rows(seal.out_path("dev"), drop_torn_tail=True)) == 1 and os.path.exists(seal.out_path("dev") + ".recovery.log"), "documented recovery drops only the torn line and logs it")
    put("dev", [row("d1", stamp), row("d2", stamp, em=False), row("d3", stamp)])
    ok(len(seal.load_complete("dev", stamp)) == 3, "a clean development file loads")
    # development provenance before selection (Lucien's recheck)
    refuses(analyze.select, "no development run manifest", "selection without a completed-run manifest")
    run_readers.main(["dev"])                                     # complete file, no model needed: records the completion hash
    ok(analyze.verified_dev()[2] == seal.sha_file(seal.out_path("dev")), "a completed, unchanged development file verifies")
    good = open(seal.out_path("dev"), "rb").read()
    put("dev", [row("d1", stamp, em=False), row("d2", stamp), row("d3", stamp)])     # valid rows, but edited after completion
    refuses(analyze.select, "does not match the hash recorded", "development output edited after completion")
    open(seal.out_path("dev"), "wb").write(good)
    analyze.select(); pk = json.load(open(T + "out/picks.json"))
    ok(pk["dev_output_sha"] == seal.sha_file(seal.out_path("dev")) and pk["inference_stamp"] == stamp, "picks bind the verified development output hash")
    # sealing
    json.dump({"inference_stamp": "stale", "picks": {}}, open(T + "out/picks.json", "w"))
    open(T + "PREREG.md", "w").write("draft\nSEALED: (not yet)\n")
    refuses(seal.make_seal, "not yet", "sealing an unsealed-looking PREREG")
    open(T + "PREREG.md", "w").write("final text\nSEALED: test\n")
    refuses(seal.make_seal, "different inference stamp", "sealing picks made under another stamp")
    json.dump({"inference_stamp": stamp, "dev_output_sha": seal.sha_file(seal.out_path("dev")),
               "picks": {rd: {"layer": 1, "measure": "p1", "direction": 1} for rd in config.READERS}}, open(T + "out/picks.json", "w"))
    seal.make_seal(); sha = seal.require_sealed()
    refuses(seal.make_seal, "already exists", "sealing twice")
    # changed script / data / prereg / picks / lens / model after sealing
    def tamper(path, what, needle="differs from SEAL.json"):
        orig = open(path, "rb").read(); open(path, "ab").write(b"\n#x")
        refuses(seal.require_sealed, needle, what); open(path, "wb").write(orig); seal.require_sealed()
    tamper(T + "run_readers.py", "changed inference script"); tamper(T + "marking.py", "changed marking")
    tamper(T + "analyze.py", "changed analysis script"); tamper(T + "PREREG.md", "changed pre-registration")
    tamper(T + "out/picks.json", "changed picks"); tamper(T + "lens/R.pt", "changed reader file")
    tamper(T + "model/weights.bin", "changed model file"); tamper(T + "data/final.jsonl", "changed final data")
    orig = open(T + "data/final.jsonl", "rb").read(); open(T + "data/final.jsonl", "ab").write(b"\n")
    refuses(lambda: seal.check_data_bytes("final"), "not the registered", "data bytes gate"); open(T + "data/final.jsonl", "wb").write(orig)
    # final resume / provenance
    put("final", [row("f1", "old-code-stamp")])
    refuses(lambda: run_readers.main(["final"]), "different stamp", "final resume into rows made before the seal")
    put("final", [row("f1", sha), row("f2", sha, em=False), row("f3", sha), row("f4", sha)])
    run_readers.main(["final"])                                   # complete file: no model needed, records the output hash
    side = json.load(open(seal.out_path("final").replace(".jsonl", ".manifest.json")))
    ok(side["stamp"] == sha and side["output_sha"] == seal.sha_file(seal.out_path("final")), "run manifest records seal and output hash")
    put("final", [row("f1", sha), row("f2", sha, em=False), row("f3", sha), row("f4", sha, em=False)])
    refuses(analyze.final, "does not match the hash recorded", "final output edited after completion")
finally:
    shutil.rmtree(T, ignore_errors=True)
print(f"ALL {N[0]} CHECKS PASSED (asserts {'OFF' if not __debug__ else 'on'})")
