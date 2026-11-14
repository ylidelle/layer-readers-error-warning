"""SPRINT settings shared by every script (2026-11-14). Changing anything here changes the inference stamp."""
BASE = "E:/OneDrive/Claude Orion Bennett/Lab/epi_sprint/"
MODEL = "E:/lab-data/models/Qwen3.5-4B"           # Qwen/Qwen3.5-4B, revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
MODEL_TAG = "qwen35_4b"
LENS_DIR = "E:/lab-data/lenses_neuronpedia/qwen3.5-4b/"
LENSES = {"J": "jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens.pt",
          "R": "rlens/Salesforce-wikitext/Qwen3.5-4B_rlens.pt",
          "J++": "jpp/Salesforce-wikitext/Qwen3.5-4B_jpp_lens.pt"}
READERS = ["logit", "J", "R", "J++"]
MEASURES = ["p1", "ent", "margin"]
LAYERS = list(range(0, 31))        # layers every reader file covers; the model's last block (31) is not read by any reader
N_MODEL_LAYERS = 32
DATA_SHA = {"dev": "2f6651f4686653bcce8ae8b0071897777f17ab5fcd2610929313325a448ffd77",
            "final": "586e55001cbac1ce5c2051b7dc08a2c21544bbeb95ae07c3ab5860a949d67040"}
N_ROWS = {"dev": 300, "final": 1000}
SUFFIX = " Answer with just the answer, no explanation."
ASK_CONF = "How confident are you that your answer is correct? Reply with a number from 0 to 100 only."
GEN = {"do_sample": False, "max_new_tokens_answer": 24, "max_new_tokens_conf": 6, "enable_thinking": False,
       "dtype": "bfloat16"}
MAX_SEQ = 4096                     # one length policy for readers and generation: longer prompts are an ERROR, never truncated
FIXED_LAYER, FIXED_MEASURE = 23, "p1"   # robustness check chosen without data; direction fixed: higher = more likely correct
MARGIN = 0.05
N_BOOT, SEED = 10000, 20261114
AUDIT_N, AUDIT_SEED = 100, 20261115
