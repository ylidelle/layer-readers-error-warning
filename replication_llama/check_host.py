"""SPRINT host check (2026-11-14; Lucien's review, points 2 and 6). Synthetic prompts only. Needs the GPU.
A. Do the four reader calls and generation see the SAME token ids (short, Unicode, and longer than 512 tokens)?
B. How do these relate at the last prompt position: a plain forward pass, generation's unprocessed logits,
   generation's processed scores, the logits the reader library returns for the model, and the logit lens at the
   model's true last block (31) and at the last layer the reader files cover (30)?"""
import sys, json, torch, jlens, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import config
sys.stdout.reconfigure(encoding="utf-8")
tok = AutoTokenizer.from_pretrained(config.MODEL)
hf = AutoModelForCausalLM.from_pretrained(config.MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
model = jlens.from_hf(hf, tok)
lenses = {k: jlens.JacobianLens.load(config.LENS_DIR + v) for k, v in config.LENSES.items()}
print("versions:", "torch", torch.__version__, "| transformers", transformers.__version__)
print("model layers:", model.n_layers, "| reader files cover:", {k: (min(l.source_layers), max(l.source_layers)) for k, l in lenses.items()})
print("bos_token:", repr(tok.bos_token), "| add_bos_token attr:", getattr(tok, "add_bos_token", "absent"))
gc = hf.generation_config
print("generation config:", {k: getattr(gc, k, None) for k in ("do_sample", "temperature", "top_p", "top_k", "repetition_penalty", "no_repeat_ngram_size", "suppress_tokens", "begin_suppress_tokens", "forced_eos_token_id")})
def chat(u): return tok.apply_chat_template([{"role": "user", "content": u}], add_generation_prompt=True, tokenize=False)
_bos = tok.bos_token or ""
def rt(p): return p[len(_bos):] if _bos and p.startswith(_bos) else p      # same rule as run_readers.reader_text
print("first ids of a templated prompt:", tok(chat("x"), add_special_tokens=False)["input_ids"][:4], "| bos id", tok.bos_token_id)
P = {"short": chat("Which planet is known as the Red Planet?" + config.SUFFIX),
     "unicode": chat("Qui a écrit « Les Misérables » ? 東京 is the capital of which country? Ünïcödé ✓" + config.SUFFIX),
     "long": chat("Background: " + "The quick brown fox jumps over the lazy dog near the river bank. " * 60 + "Question: what animal jumps?" + config.SUFFIX)}
res = {}
for name, p in P.items():
    ins = tok(p, return_tensors="pt", add_special_tokens=False).to("cuda")
    gen_ids = ins["input_ids"][0].cpu(); n = gen_ids.shape[0]
    same = {}
    for rd in config.READERS:
        lens = lenses["J"] if rd == "logit" else lenses[rd]
        lg, mlog, ids = lens.apply(model, rt(p), layers=config.LAYERS, positions=[-1], max_seq_len=config.MAX_SEQ, use_jacobian=(rd != "logit"))
        same[rd] = bool(torch.equal(ids[0].cpu(), gen_ids))
    _, _, ids512 = lenses["J"].apply(model, rt(p), layers=[30], positions=[-1], use_jacobian=False)       # library default length
    with torch.no_grad():
        fwd = hf(**ins).logits[0, -1].float().cpu()
        o = hf.generate(**ins, max_new_tokens=4, do_sample=False, output_scores=True, output_logits=True, return_dict_in_generate=True)
    g_log, g_sc = o.logits[0][0].float().cpu(), o.scores[0][0].float().cpu()
    first = int(o.sequences[0, n])
    l31, mlog, _ = lenses["J"].apply(model, rt(p), layers=[31], positions=[-1], max_seq_len=config.MAX_SEQ, use_jacobian=False)
    l30, _, _ = lenses["J"].apply(model, rt(p), layers=[30], positions=[-1], max_seq_len=config.MAX_SEQ, use_jacobian=False)
    mlog, l31v, l30v = mlog[-1], l31[31][-1], l30[30][-1]
    d = lambda a, b: float((a - b).abs().max())
    sm = lambda v: float(torch.softmax(v, -1).max())
    r = {"n_tokens": n, "ids_equal_all_readers": same, "library_default_would_truncate_to": int(ids512.shape[1]),
         "first_token": first,
         "argmax": {"forward": int(fwd.argmax()), "gen_logits": int(g_log.argmax()), "gen_scores": int(g_sc.argmax()),
                    "reader_model_logits": int(mlog.argmax()), "logit_lens_L31": int(l31v.argmax()), "logit_lens_L30": int(l30v.argmax())},
         "top1_prob": {"forward": sm(fwd), "gen_logits": sm(g_log), "gen_scores": sm(g_sc), "reader_model_logits": sm(mlog),
                       "logit_lens_L31": sm(l31v), "logit_lens_L30": sm(l30v)},
         "max_abs_diff": {"gen_scores vs gen_logits": d(g_sc, g_log), "gen_logits vs forward": d(g_log, fwd),
                          "reader_model_logits vs forward": d(mlog, fwd), "logit_lens_L31 vs reader_model_logits": d(l31v, mlog),
                          "logit_lens_L30 vs forward": d(l30v, fwd)},
         "scores_have_minus_inf": bool(torch.isinf(g_sc).any())}
    res[name] = r
    print(f"\n== {name}: {n} tokens"); print(json.dumps(r, indent=1))
json.dump(res, open(config.BASE + "out/host_check.json", "w", encoding="utf-8"), indent=1)
ok = all(all(r["ids_equal_all_readers"].values()) for r in res.values())
print("\nTOKEN IDS IDENTICAL FOR EVERY READER ON EVERY PROMPT:", ok)
sys.exit(0 if ok else 1)
