#!/bin/sh
# Runs once on the rented machine. Installs the same library versions as the primary study, downloads the model,
# the three reader files and the TriviaQA file, and rebuilds the question sets. The Hugging Face token comes from
# the environment (a RunPod secret) and is never printed.
set -e
cd /workspace
pip install -q "transformers==5.14.1" "tokenizers==0.22.2" "accelerate==1.14.0" "safetensors==0.8.0" "huggingface_hub==1.26.0" pyarrow \
  "jlens @ git+https://github.com/anthropics/jacobian-lens@581d398613e5602a5af361e1c34d3a92ea82ba8e"
python - <<'PY'
import os
from huggingface_hub import snapshot_download, hf_hub_download
assert os.environ.get("HF_TOKEN"), "HF_TOKEN is not set"
snapshot_download("meta-llama/Llama-3.1-8B-Instruct", revision="0e9e39f249a16976918f6564b8830bc894c89659",
                  allow_patterns=["*.json", "*.safetensors", "tokenizer*"], local_dir="/workspace/models/Llama-3.1-8B-Instruct")
for f in ("jlens/Salesforce-wikitext/Llama-3.1-8B-Instruct_jacobian_lens.pt", "rlens/Salesforce-wikitext/Llama-3.1-8B-Instruct_rlens.pt",
          "jpp/Salesforce-wikitext/Llama-3.1-8B-Instruct_jpp_lens.pt"):
    hf_hub_download("neuronpedia/jacobian-lens", "llama3.1-8b-it/" + f, local_dir="/workspace/lenses")
print("downloads done")
PY
rm -rf /workspace/models/Llama-3.1-8B-Instruct/.cache /workspace/lenses/.cache
mkdir -p /workspace/data /workspace/epi/data /workspace/epi/out
curl -sL -o /workspace/data/rc.nocontext_validation_0.parquet "https://huggingface.co/api/datasets/mandarjoshi/trivia_qa/parquet/rc.nocontext/validation/0.parquet"
sha256sum /workspace/data/rc.nocontext_validation_0.parquet
cd /workspace/epi && python build_sets.py | grep -v "dev example" && ls -la /workspace/models/Llama-3.1-8B-Instruct /workspace/lenses/llama3.1-8b-it/*/*/ && echo BOOTSTRAP DONE
