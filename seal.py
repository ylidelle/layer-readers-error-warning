"""SPRINT integrity layer (2026-11-14, after Lucien's pre-seal review, point 1).

Two stamps:
  inference stamp  = hash of everything that can change a row of output: settings, the inference and marking code,
                     both question files, the three reader files, every model file, package versions.
  seal             = SEAL.json: the inference stamp plus hashes of PREREG.md, picks.json, analyze.py, build_sets.py.

  python seal.py stamp     print the current inference stamp
  python seal.py make      write SEAL.json (refuses if one exists, or if the picks were made under another stamp)
  python seal.py check     compare the current state with SEAL.json

Every gate raises Refused (a real exception, not an assert, so `python -O` cannot switch it off)."""
import sys, os, json, hashlib, math, importlib.metadata as md
import config

class Refused(SystemExit):
    def __init__(self, msg):
        super().__init__("REFUSED: " + msg)

def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()

def sha_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

def _versions():
    out = {"python": sys.version.split()[0]}
    for p in ("torch", "transformers", "tokenizers", "jlens", "numpy"):
        try: out[p] = md.version(p)
        except md.PackageNotFoundError: out[p] = "not installed"
    return out

def inference_manifest():
    B = config.BASE
    settings = {k: getattr(config, k) for k in ("MODEL_TAG", "LENSES", "READERS", "MEASURES", "LAYERS", "N_MODEL_LAYERS",
                                                "DATA_SHA", "N_ROWS", "SUFFIX", "ASK_CONF", "GEN", "MAX_SEQ")}
    return {"settings": settings,
            "code": {n: sha_file(B + n) for n in ("config.py", "seal.py", "run_readers.py", "marking.py")},
            "data": {s: sha_file(B + f"data/{s}.jsonl") for s in ("dev", "final")},
            "lenses": {k: sha_file(config.LENS_DIR + v) for k, v in config.LENSES.items()},
            "model": {n: sha_file(os.path.join(config.MODEL, n)) for n in sorted(os.listdir(config.MODEL))
                      if os.path.isfile(os.path.join(config.MODEL, n))},
            "versions": _versions()}

def inference_stamp(man=None):
    return sha_obj(man or inference_manifest())

def check_data_bytes(name):
    """Hash the question file WITHOUT parsing or showing it."""
    got = sha_file(config.BASE + f"data/{name}.jsonl")
    if got != config.DATA_SHA[name]:
        raise Refused(f"data/{name}.jsonl hash {got[:12]} is not the registered {config.DATA_SHA[name][:12]}")

def full_manifest():
    B = config.BASE
    try: picks = json.load(open(B + "out/picks.json", encoding="utf-8"))
    except Exception: picks = {"inference_stamp": "picks.json unreadable"}
    inf = inference_manifest()
    return {"inference": inf, "inference_stamp": inference_stamp(inf),
            "prereg_sha": sha_file(B + "PREREG.md"), "picks_sha": sha_file(B + "out/picks.json"),
            "picks_made_under_stamp": picks.get("inference_stamp"),
            "analysis_code": {n: sha_file(B + n) for n in ("analyze.py", "build_sets.py")}}

def make_seal():
    B = config.BASE
    if os.path.exists(B + "SEAL.json"): raise Refused("SEAL.json already exists; a seal is made once")
    if "SEALED: (not yet)" in open(B + "PREREG.md", encoding="utf-8").read():
        raise Refused("PREREG.md still says 'SEALED: (not yet)'")
    man = full_manifest()
    if man["picks_made_under_stamp"] != man["inference_stamp"]:
        raise Refused("picks.json was made under a different inference stamp than the current one")
    blob = json.dumps(man, sort_keys=True, indent=1, ensure_ascii=False).encode("utf-8")
    with open(B + "SEAL.json", "xb") as f: f.write(blob)
    print("SEAL.json written. sha256:", hashlib.sha256(blob).hexdigest())
    print("PREREG.md sha256:", man["prereg_sha"])

def require_sealed():
    """Returns the seal's sha256. Raises unless the current state equals SEAL.json in every entry."""
    B = config.BASE
    if not os.path.exists(B + "SEAL.json"): raise Refused("no SEAL.json: the pre-registration is not sealed")
    sealed = json.load(open(B + "SEAL.json", encoding="utf-8"))
    now = full_manifest()
    def diff(a, b, path=""):
        out = []
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)): out += diff(a.get(k), b.get(k), path + "/" + str(k))
        elif a != b: out.append(path)
        return out
    d = diff(sealed, now)
    if d: raise Refused("current state differs from SEAL.json at: " + ", ".join(d[:8]))
    return sha_file(B + "SEAL.json")

def out_path(name):
    return config.BASE + f"out/{name}_{config.MODEL_TAG}.jsonl"

def expected_qids(name):
    return [json.loads(l)["qid"] for l in open(config.BASE + f"data/{name}.jsonl", encoding="utf-8")]

def read_rows(path, drop_torn_tail=False):
    """Rows of an output file. A last line that is not valid JSON (a write cut off by a crash) is an ERROR unless
    drop_torn_tail is set; then that one line is removed from the file and the removal is logged beside it."""
    if not os.path.exists(path): return []
    raw = open(path, "rb").read()
    lines = raw.split(b"\n")
    if lines and lines[-1] == b"": lines.pop()
    rows = []
    for i, ln in enumerate(lines):
        try: rows.append(json.loads(ln.decode("utf-8")))
        except Exception:
            if i == len(lines) - 1 and drop_torn_tail:
                open(path, "wb").write(b"".join(l + b"\n" for l in lines[:-1]))
                with open(path + ".recovery.log", "a", encoding="utf-8") as f:
                    f.write(f"dropped a torn last line of {len(ln)} bytes (line {i+1})\n")
                break
            raise Refused(f"{os.path.basename(path)} line {i+1} is not valid JSON"
                          + (" (torn tail: rerun with --drop-torn-tail)" if i == len(lines) - 1 else ""))
    return rows

def check_resume(rows, name, stamp):
    """Existing rows must carry this run's stamp and be a duplicate-free PREFIX of the expected question order."""
    exp = expected_qids(name)
    got = [r.get("qid") for r in rows]
    if len(set(got)) != len(got): raise Refused("existing output has duplicate question ids")
    if got != exp[:len(got)]: raise Refused("existing output is not a prefix of the expected question order")
    bad = sorted({str(r.get("stamp"))[:12] for r in rows if r.get("stamp") != stamp})
    if bad: raise Refused(f"existing rows were made under a different stamp ({', '.join(bad)}); cannot resume into them")
    return exp[len(got):]

def load_complete(name, stamp):
    """All rows of a finished run, verified: stamp, exact question ids once each, schema, finite numbers, both classes."""
    rows = read_rows(out_path(name))
    exp = expected_qids(name)
    if len(exp) != config.N_ROWS[name]: raise Refused(f"data/{name}.jsonl has {len(exp)} questions, expected {config.N_ROWS[name]}")
    got = [r.get("qid") for r in rows]
    if len(got) != len(set(got)): raise Refused(f"{name} output has duplicate question ids")
    if set(got) != set(exp): raise Refused(f"{name} output has {len(got)} rows; ids differ from the {len(exp)} expected "
                                           f"(missing {len(set(exp)-set(got))}, unexpected {len(set(got)-set(exp))})")
    for r in rows:
        if r.get("stamp") != stamp: raise Refused(f"row {r['qid']} carries stamp {str(r.get('stamp'))[:12]}, expected {stamp[:12]}")
        for key in ("correct_em", "correct_contain"):
            if not isinstance(r.get(key), bool): raise Refused(f"row {r['qid']}: {key} is not true/false")
        if not (isinstance(r.get("p_out"), float) and math.isfinite(r["p_out"])): raise Refused(f"row {r['qid']}: p_out not finite")
        if r.get("said") is not None and not (0 <= r["said"] <= 100): raise Refused(f"row {r['qid']}: said out of range")
        if set(r.get("read", {})) != set(config.READERS): raise Refused(f"row {r['qid']}: reader set differs")
        for rd in config.READERS:
            if set(r["read"][rd]) != {str(L) for L in config.LAYERS}: raise Refused(f"row {r['qid']}: {rd} layer set differs")
            for L, d in r["read"][rd].items():
                for m in config.MEASURES:
                    if not (isinstance(d.get(m), float) and math.isfinite(d[m])): raise Refused(f"row {r['qid']}: {rd} L{L} {m} not finite")
    y = [r["correct_em"] for r in rows]
    if all(y) or not any(y): raise Refused(f"{name}: only one outcome class present")
    return rows

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "stamp": print(inference_stamp())
    elif cmd == "make": make_seal()
    elif cmd == "check": print("state matches SEAL.json, sha256", require_sealed())
    else: sys.exit("usage: seal.py stamp|make|check")
