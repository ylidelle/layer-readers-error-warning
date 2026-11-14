"""SPRINT inference (2026-11-14; rewritten after Lucien's pre-seal review). One pass over a question set with Qwen3.5-4B.

For each question, at the LAST prompt position (before the model writes anything), four readers are applied at
layers 0..30: logit lens, J-lens, R-lens, J++.  Stored per reader and layer (none uses the answer key or the answer):
    p1 top-1 probability · ent entropy in nats · margin top-1 minus top-2 logit · top the top-1 token id
Then the model answers (greedy) and two baselines are stored:
    p_out  probability of its own first answer token, from the PROCESSED generation scores
           (p_out_raw: the same from the unprocessed logits)
    said   stated confidence 0..100 asked in a follow-up turn, AFTER the answer (None if the reply is not one number)
The answer key is used only by marking.py to label the answer.

GATES, all before any question is parsed or the model is loaded:
  - `final` requires a SEAL.json that matches the current state in every entry; rows are stamped with the seal hash.
  - `dev` rows are stamped with the current inference stamp.
  - the question file's bytes must hash to the registered value;
  - resume only into rows with the same stamp that are a duplicate-free prefix of the expected order.
On every row the token ids seen by all four reader calls and by generation are asserted identical; a prompt longer
than MAX_SEQ is an error (nothing is ever truncated).

usage: run_readers.py <dev|final> [--limit N (dev only)] [--drop-torn-tail]"""
import sys, json, time, hashlib, os
import config, seal, marking
sys.stdout.reconfigure(encoding="utf-8")

def main(argv):
    name = argv[0] if argv else ""
    if name not in ("dev", "final"): raise seal.Refused("set must be dev or final")
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else None
    if limit and name == "final": raise seal.Refused("--limit is not allowed on the final set")
    t0 = time.time()
    man = seal.inference_manifest()
    stamp = seal.require_sealed() if name == "final" else seal.inference_stamp(man)
    seal.check_data_bytes(name)
    out = seal.out_path(name)
    rows = seal.read_rows(out, drop_torn_tail="--drop-torn-tail" in argv)
    todo_ids = seal.check_resume(rows, name, stamp)
    if limit: todo_ids = todo_ids[:max(0, limit - len(rows))]
    print(f"set {name} | stamp {stamp[:16]} | {len(rows)} rows present | {len(todo_ids)} to do | gates passed in {time.time()-t0:.0f}s", flush=True)
    side = out.replace(".jsonl", ".manifest.json")
    if not os.path.exists(side):
        json.dump({"set": name, "stamp": stamp, "inference_manifest": man, "started": time.strftime("%Y-%m-%d %H:%M:%S")},
                  open(side, "w", encoding="utf-8"), indent=1, sort_keys=True)
    elif json.load(open(side, encoding="utf-8"))["stamp"] != stamp:
        raise seal.Refused("the run's manifest file carries a different stamp")
    if not todo_ids:
        finish(out, side); return

    import torch, jlens
    from transformers import AutoTokenizer, AutoModelForCausalLM
    by_id = {}
    for l in open(config.BASE + f"data/{name}.jsonl", encoding="utf-8"):
        q = json.loads(l); by_id[q["qid"]] = q
    tok = AutoTokenizer.from_pretrained(config.MODEL)
    hf = AutoModelForCausalLM.from_pretrained(config.MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    model = jlens.from_hf(hf, tok)
    if model.n_layers != config.N_MODEL_LAYERS: raise seal.Refused(f"model has {model.n_layers} layers, expected {config.N_MODEL_LAYERS}")
    lenses = {k: jlens.JacobianLens.load(config.LENS_DIR + v) for k, v in config.LENSES.items()}
    for k, l in lenses.items():
        if not set(config.LAYERS) <= set(l.source_layers): raise seal.Refused(f"reader {k} does not cover layers 0..30")
    gc = hf.generation_config
    print(f"loaded in {time.time()-t0:.0f}s | generation config: do_sample={gc.do_sample} temperature={gc.temperature} "
          f"top_p={gc.top_p} top_k={gc.top_k} repetition_penalty={gc.repetition_penalty} | "
          + " ".join(f"{k}:n_prompts={l.n_prompts}" for k, l in lenses.items()), flush=True)

    def chat(msgs):
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=config.GEN["enable_thinking"])

    def gen(prompt, n):
        ins = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        if ins["input_ids"].shape[1] > config.MAX_SEQ: raise seal.Refused("prompt longer than MAX_SEQ")
        with torch.no_grad():
            o = hf.generate(**ins, max_new_tokens=n, do_sample=False, output_scores=True, output_logits=True,
                            return_dict_in_generate=True)
        new = o.sequences[0, ins["input_ids"].shape[1]:]
        text = tok.decode(new, skip_special_tokens=True).strip()
        return text, o.scores[0][0].float().cpu(), o.logits[0][0].float().cpu(), int(new[0]), ins["input_ids"][0].cpu()

    def stats(v):
        v = v.float(); lp = torch.log_softmax(v, -1); p = lp.exp(); top = torch.topk(v, 2)
        return {"p1": float(p.max()), "ent": float(-(p * lp).sum()), "margin": float(top.values[0] - top.values[1]),
                "top": int(top.indices[0])}

    f = open(out, "a", encoding="utf-8")
    for i, qid in enumerate(todo_ids):
        q = by_id[qid]
        p = chat([{"role": "user", "content": q["question"] + config.SUFFIX}])
        ans, scores, logits, first_id, gen_ids = gen(p, config.GEN["max_new_tokens_answer"])
        if int(scores.argmax()) != first_id: raise seal.Refused(f"{qid}: first generated token is not the top processed score")
        read, fwd_top = {}, None
        for rd in config.READERS:
            lens = lenses["J"] if rd == "logit" else lenses[rd]
            lg, model_logits, ids = lens.apply(model, p, layers=config.LAYERS, positions=[-1],
                                               max_seq_len=config.MAX_SEQ, use_jacobian=(rd != "logit"))
            if not torch.equal(ids[0].cpu(), gen_ids):
                raise seal.Refused(f"{qid}: reader {rd} saw different token ids from generation "
                                   f"({ids.shape[1]} vs {gen_ids.shape[0]} tokens)")
            read[rd] = {str(L): stats(lg[L][-1]) for L in config.LAYERS}
            fwd_top = int(model_logits[-1].argmax())
        p_out = float(torch.softmax(scores, -1).max())
        p_out_raw = float(torch.softmax(logits, -1)[first_id])
        p2 = chat([{"role": "user", "content": q["question"] + config.SUFFIX}, {"role": "assistant", "content": ans},
                   {"role": "user", "content": config.ASK_CONF}])
        ctext, _, _, _, _ = gen(p2, config.GEN["max_new_tokens_conf"])
        row = {"qid": qid, "stamp": stamp, "answer": ans,
               "correct_em": marking.exact_match(ans, q["aliases"]), "correct_contain": marking.contains_alias(ans, q["aliases"]),
               "p_out": p_out, "p_out_raw": p_out_raw, "first_id": first_id, "fwd_top": fwd_top,
               "said": marking.parse_confidence(ctext), "said_raw": ctext,
               "n_tok": int(gen_ids.shape[0]), "ids_sha": hashlib.sha256(gen_ids.numpy().tobytes()).hexdigest()[:16], "read": read}
        f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
        if (name == "dev" and (i < 10 or (i + 1) % 25 == 0)) or (name == "final" and (i + 1) % 50 == 0):
            shown = f"{q['question'][:50]!r} -> {ans[:30]!r}" if name == "dev" else "(final: text not shown)"
            print(f"{len(rows)+i+1:4d} em={int(row['correct_em'])} contain={int(row['correct_contain'])} p_out={p_out:.2f} "
                  f"said={row['said']} {time.time()-t0:.0f}s {shown}", flush=True)
    f.close()
    print(f"appended {len(todo_ids)} rows in {time.time()-t0:.0f}s | peak VRAM {torch.cuda.max_memory_allocated()/2**30:.1f} GB")
    finish(out, side)

def finish(out, side):
    """When the file is complete, write its hash into the run's manifest file (the raw output is hashed before analysis)."""
    name = "final" if "final_" in os.path.basename(out) else "dev"
    n = len(seal.read_rows(out))
    if n == config.N_ROWS[name]:
        s = json.load(open(side, encoding="utf-8"))
        h = seal.sha_file(out)
        if s.get("output_sha") not in (None, h): raise seal.Refused("complete output no longer matches the hash recorded at completion")
        s["output_sha"], s["rows"] = h, n
        s.setdefault("completed", time.strftime("%Y-%m-%d %H:%M:%S"))
        json.dump(s, open(side, "w", encoding="utf-8"), indent=1, sort_keys=True)
        print(f"COMPLETE: {n} rows, output sha256 {h}")
    else:
        print(f"not complete: {n} of {config.N_ROWS[name]} rows")

if __name__ == "__main__":
    main(sys.argv[1:])
