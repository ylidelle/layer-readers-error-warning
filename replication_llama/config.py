"""SECONDARY REPLICATION on Llama 3.1 8B (written 2026-11-15, after the Qwen results were seen).
Settings shared by every script. Paths are those of the rented machine. Changing anything here changes the stamp."""
BASE = "/workspace/epi/"
MODEL = "/workspace/models/Llama-3.1-8B-Instruct"     # meta-llama/Llama-3.1-8B-Instruct, snapshot 0e9e39f249a16976918f6564b8830bc894c89659
MODEL_TAG = "llama31_8b"
LENS_DIR = "/workspace/lenses/llama3.1-8b-it/"        # neuronpedia/jacobian-lens, folder llama3.1-8b-it
LENSES = {"J": "jlens/Salesforce-wikitext/Llama-3.1-8B-Instruct_jacobian_lens.pt",
          "R": "rlens/Salesforce-wikitext/Llama-3.1-8B-Instruct_rlens.pt",
          "J++": "jpp/Salesforce-wikitext/Llama-3.1-8B-Instruct_jpp_lens.pt"}
READERS = ["logit", "J", "R", "J++"]
MEASURES = ["p1", "ent", "margin"]
LAYERS = list(range(0, 31))        # layers every reader file covers; the model's last block (31) is read by no reader
N_MODEL_LAYERS = 32
DATA_SHA = {"dev": "2f6651f4686653bcce8ae8b0071897777f17ab5fcd2610929313325a448ffd77",      # the SAME question files as the Qwen study
            "final": "586e55001cbac1ce5c2051b7dc08a2c21544bbeb95ae07c3ab5860a949d67040"}
N_ROWS = {"dev": 300, "final": 1000}
SUFFIX = " Answer with just the answer, no explanation."
ASK_CONF = "How confident are you that your answer is correct? Reply with a number from 0 to 100 only."
GEN = {"do_sample": False, "max_new_tokens_answer": 24, "max_new_tokens_conf": 6, "enable_thinking": None,   # None: the template is called with no extra switch
       "dtype": "bfloat16"}
MAX_SEQ = 4096
FIXED_LAYER, FIXED_MEASURE = 23, "p1"
MARGIN = 0.05
N_BOOT, SEED = 10000, 20261114
AUDIT_N, AUDIT_SEED = 100, 20261115
