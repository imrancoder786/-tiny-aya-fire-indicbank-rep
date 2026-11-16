# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "vllm>=0.6.3",
#     "torch",
#     "transformers",
#     "accelerate",
#     "huggingface_hub>=0.34.0",
#     "PyYAML>=6.0",
#     "openai>=1.40.0",
#     "rich>=13.0",
#     "tqdm>=4.66.3",
#     "python-dotenv>=1.0.0",
#     "pydantic>=2.0",
#     "httpx",
#     "tenacity",
#     "jinja2",
#     "matplotlib",
#     "pandas",
#     "numpy",
#     "pip",
#     "setuptools",
# ]
# [tool.hf-jobs]
# flavor = "a100-large"
# timeout = "2h"
# secrets = ["HF_TOKEN"]
# ///

import os
import sys
import json
import time
import yaml
import shutil
import requests
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import torch
from huggingface_hub import login, snapshot_download, HfApi

# ==============================================================================
# 1. Configuration & Authentication
# ==============================================================================
CANDIDATE_REPO = os.environ.get("CANDIDATE_REPO", "imrancoder/tiny-aya-fire-indicbank")
JUDGE_REPO = os.environ.get("JUDGE_REPO", "Qwen/Qwen2.5-7B-Instruct-AWQ")
DATASET_REPO = "NPCI/IndicBankBench"
RUN_ID = "tiny_aya_fire_indicbank_ft"

WORK_DIR = Path("/tmp/indicbank_eval")
REPO_ROOT = WORK_DIR / "IndicBankBench"
DATA_DIR = REPO_ROOT / "data"
CANDIDATE_LOCAL_DIR = WORK_DIR / "candidate_model_16k"

WORK_DIR.mkdir(parents=True, exist_ok=True)

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise RuntimeError("HF_TOKEN environment variable is missing! Pass `-s HF_TOKEN` to `hf jobs`.")

login(token=hf_token)
api = HfApi(token=hf_token)
print(f"[1/8] Authenticated with Hugging Face. Evaluating: {CANDIDATE_REPO}")

# ==============================================================================
# 2. Clone & Install IndicBankBench
# ==============================================================================
if REPO_ROOT.exists():
    shutil.rmtree(REPO_ROOT)

subprocess.run(
    ["git", "clone", "https://github.com/npci/IndicBankBench.git", str(REPO_ROOT)],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO_ROOT)],
    check=True,
)
print("[2/8] Cloned and installed IndicBankBench.")

# ==============================================================================
# 3. Download Dataset, Fine-Tuned Candidate Model & Judge Model
# ==============================================================================
print(f"[3/8] Downloading dataset {DATASET_REPO}...")
snapshot_download(
    repo_id=DATASET_REPO,
    repo_type="dataset",
    local_dir=str(DATA_DIR),
    token=hf_token,
)
os.environ["INDICBANKBENCH_DATA"] = str(DATA_DIR)

print(f"[3/8] Downloading fine-tuned model {CANDIDATE_REPO}...")
snapshot_download(
    repo_id=CANDIDATE_REPO,
    local_dir=str(CANDIDATE_LOCAL_DIR),
    token=hf_token,
    ignore_patterns=["ckpt/*", "eval_results/*"],  # skip training checkpoints (adapters/optimizer states)
)

# The merged model was saved with transformers 5, whose config/tokenizer files use new field names
# (rope_parameters, tokenizer_class=TokenizersBackend) that the transformers 4.x bundled with vLLM
# cannot parse (and a missing rope_theta would silently fall back to the wrong RoPE base).
# The architecture is identical to the base model, so reuse the BASE config/tokenizer files in the
# classic format and only raise max_position_embeddings to the 16384 eval length.
from huggingface_hub import hf_hub_download
BASE_REPO = "CohereLabs/tiny-aya-fire"
cfg_path = CANDIDATE_LOCAL_DIR / "config.json"
cfg_data = json.loads(Path(hf_hub_download(BASE_REPO, "config.json", token=hf_token)).read_text())
cfg_data["max_position_embeddings"] = max(16384, int(cfg_data.get("max_position_embeddings", 8192)))
cfg_path.write_text(json.dumps(cfg_data, indent=2))
for fn in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"):
    try:
        shutil.copy(hf_hub_download(BASE_REPO, fn, token=hf_token), CANDIDATE_LOCAL_DIR / fn)
    except Exception as e:
        print(f"(base {fn} not copied: {e})")
for stale in ("chat_template.jinja", "cohere_tool_template.jinja"):
    (CANDIDATE_LOCAL_DIR / stale).unlink(missing_ok=True)  # the explicit --chat-template below is used
print("Rewrote candidate config.json + tokenizer files in transformers-4 format (max_position_embeddings=16384).")

print(f"[3/8] Pre-caching Judge model {JUDGE_REPO}...")
snapshot_download(repo_id=JUDGE_REPO, token=hf_token)

# ==============================================================================
# 4. Patch Token Limits (16384 -> 1536) & Install sitecustomize.py Hook
# ==============================================================================
models_yaml = REPO_ROOT / "indicbankbench" / "config" / "models.yaml"
for ext in ("*.py", "*.yaml", "*.yml", "*.json"):
    for fpath in REPO_ROOT.rglob(ext):
        if "results" in fpath.parts or "data" in fpath.parts or ".git" in fpath.parts:
            continue
        txt = fpath.read_text(encoding="utf-8", errors="ignore")
        if "16384" in txt or "16_384" in txt:
            new_txt = txt.replace("16384", "1536").replace("16_384", "1536")
            fpath.write_text(new_txt, encoding="utf-8")

def patch_dict_tokens(d):
    if isinstance(d, dict):
        for k, v in list(d.items()):
            if k in ("max_tokens", "max_completion_tokens") and isinstance(v, int) and v > 1536:
                d[k] = 1536
            else:
                patch_dict_tokens(v)
    elif isinstance(d, list):
        for item in d:
            patch_dict_tokens(item)

cfg = yaml.safe_load(models_yaml.read_text()) or {}
patch_dict_tokens(cfg)
models_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False))

sitecustomize_code = """import openai.resources.chat.completions as _cc

_orig_create = _cc.Completions.create
def _safe_create(self, *args, **kwargs):
    if kwargs.get("max_tokens") is not None and kwargs["max_tokens"] > 1536:
        kwargs["max_tokens"] = 1536
    if kwargs.get("max_completion_tokens") is not None and kwargs["max_completion_tokens"] > 1536:
        kwargs["max_completion_tokens"] = 1536
    if "max_tokens" not in kwargs and "max_completion_tokens" not in kwargs:
        kwargs["max_tokens"] = 1536
    return _orig_create(self, *args, **kwargs)
_cc.Completions.create = _safe_create

if hasattr(_cc, "AsyncCompletions"):
    _orig_acreate = _cc.AsyncCompletions.create
    async def _safe_acreate(self, *args, **kwargs):
        if kwargs.get("max_tokens") is not None and kwargs["max_tokens"] > 1536:
            kwargs["max_tokens"] = 1536
        if kwargs.get("max_completion_tokens") is not None and kwargs["max_completion_tokens"] > 1536:
            kwargs["max_completion_tokens"] = 1536
        if "max_tokens" not in kwargs and "max_completion_tokens" not in kwargs:
            kwargs["max_tokens"] = 1536
        return await _orig_acreate(self, *args, **kwargs)
    _cc.AsyncCompletions.create = _safe_acreate
"""
(REPO_ROOT / "sitecustomize.py").write_text(sitecustomize_code)
(REPO_ROOT / "indicbankbench" / "sitecustomize.py").write_text(sitecustomize_code)

env_content = f"""JUDGE_BASE_URL=http://127.0.0.1:8001/v1
JUDGE_MODEL=judge-model
JUDGE_API_KEY=EMPTY
CANDIDATE_BASE_URL=http://127.0.0.1:8000/v1
CANDIDATE_MODEL=tiny-aya-fire-indicbank
CANDIDATE_API_KEY=EMPTY
INDICBANKBENCH_DATA={DATA_DIR}
"""
(REPO_ROOT / ".env").write_text(env_content)
(REPO_ROOT / "indicbankbench" / ".env").write_text(env_content)
for line in env_content.strip().splitlines():
    k, v = line.split("=", 1)
    os.environ[k] = v

print("[4/8] Patched models.yaml and installed OpenAI max_tokens=1536 clamp.")

# ==============================================================================
# 5. Launch Candidate & Judge vLLM Servers (Auto-Adapts to 1-GPU or Multi-GPU)
# ==============================================================================
num_gpus = torch.cuda.device_count()
if num_gpus == 0:
    raise RuntimeError("No GPU detected! Make sure to pass `--flavor a100-large` or `--flavor a10g-largex2`.")

total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
print(f"[5/8] Detected {num_gpus} GPU(s). GPU 0 VRAM: {total_vram_gb:.1f} GB")

if num_gpus >= 2:
    cand_gpu, judge_gpu = "0", "1"
    cand_mem, judge_mem = "0.88", "0.88"
    concurrency = 16
else:
    # Single large GPU (e.g., A100 80GB or L40S 48GB): co-locate both servers safely
    cand_gpu, judge_gpu = "0", "0"
    cand_mem, judge_mem = "0.44", "0.44"
    concurrency = 24 if total_vram_gb >= 70 else (16 if total_vram_gb >= 45 else 8)

TOOL_CHAT_TEMPLATE = WORK_DIR / "cohere_tool_template.jinja"
TOOL_CHAT_TEMPLATE.write_text("""{{ bos_token }}{% if messages[0]['role'] == 'system' %}<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{{ messages[0]['content'] or '' }}{% if tools %}\n\n# Available Tools\nYou may call one or more functions to assist with the user query. You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{% for tool in tools %}{{ tool | tojson }}\n{% endfor %}</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>{% endif %}<|END_OF_TURN_TOKEN|>{% set loop_messages = messages[1:] %}{% else %}{% set loop_messages = messages %}{% if tools %}<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|># Available Tools\nYou may call one or more functions to assist with the user query. You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{% for tool in tools %}{{ tool | tojson }}\n{% endfor %}</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|END_OF_TURN_TOKEN|>{% endif %}{% endif %}{% for message in loop_messages %}{% if message['role'] == 'user' %}<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{{ message['content'] or '' }}<|END_OF_TURN_TOKEN|>{% elif message['role'] == 'assistant' %}<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{% if message.get('content') %}{{ message['content'] }}{% endif %}{% if message.get('tool_calls') %}{% for tc in message['tool_calls'] %}<tool_call>\n{"name": "{{ tc['function']['name'] }}", "arguments": {{ tc['function']['arguments'] if tc['function']['arguments'] is string else (tc['function']['arguments'] | tojson) }}}\n</tool_call>{% endfor %}{% endif %}<|END_OF_TURN_TOKEN|>{% elif message['role'] == 'tool' %}<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><tool_response>\n{{ message['content'] or '' }}\n</tool_response><|END_OF_TURN_TOKEN|>{% elif message['role'] == 'system' %}<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{{ message['content'] or '' }}<|END_OF_TURN_TOKEN|>{% endif %}{% endfor %}<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>""")

env_cand = os.environ.copy()
env_cand["CUDA_VISIBLE_DEVICES"] = cand_gpu
env_cand["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
env_cand["VLLM_USE_FLASHINFER_SAMPLER"] = "0"  # FlashInfer sampler JIT-compiles with nvcc, which this image lacks

cand_log = WORK_DIR / "vllm_candidate.log"
candidate_cmd = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", str(CANDIDATE_LOCAL_DIR),
    "--served-model-name", "tiny-aya-fire-indicbank",
    "--dtype", "bfloat16",  # model was trained/merged in bf16; A100 supports it natively
    "--max-model-len", "16384",
    "--gpu-memory-utilization", cand_mem,
    "--enforce-eager",
    "--enable-auto-tool-choice",
    "--tool-call-parser", "hermes",
    "--chat-template", str(TOOL_CHAT_TEMPLATE),
    "--port", "8000",
    "--trust-remote-code",
]
candidate_proc = subprocess.Popen(
    candidate_cmd, env=env_cand, stdout=open(cand_log, "w"), stderr=subprocess.STDOUT
)

def wait_for_server(url: str, name: str, log_path: Path, proc=None, timeout: int = 900):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{url}/v1/models", timeout=5)
            if r.status_code == 200:
                print(f"[READY] {name} is live at {url}/v1")
                return
        except Exception:
            pass
        if proc is not None and proc.poll() is not None:
            break  # server process already died - stop waiting
        time.sleep(5)
    txt = log_path.read_text(errors="ignore")
    print(f"=== ERROR LOG FOR {name} (first error lines, then tail) ===")
    err_lines = [l for l in txt.splitlines() if any(k in l for k in ("Error", "ERROR", "error:", "Exception", "ValueError", "KeyError", "OOM", "out of memory"))]
    print("\n".join(err_lines[:60]))
    print("--- tail ---")
    print(txt[-6000:])
    raise RuntimeError(f"Timeout/failure waiting for {name}")

wait_for_server("http://127.0.0.1:8000", f"Candidate ({CANDIDATE_REPO})", cand_log, candidate_proc)
# Start the judge only after the candidate has finished memory profiling (avoids a startup race on a shared GPU).
env_judge = os.environ.copy()
env_judge["CUDA_VISIBLE_DEVICES"] = judge_gpu
env_judge["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

judge_log = WORK_DIR / "vllm_judge.log"
judge_cmd = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", JUDGE_REPO,
    "--served-model-name", "judge-model",
    "--quantization", "awq",
    "--dtype", "half",
    "--max-model-len", "16384",
    "--gpu-memory-utilization", judge_mem,
    "--enforce-eager",
    "--port", "8001",
    "--trust-remote-code",
]
judge_proc = subprocess.Popen(
    judge_cmd, env=env_judge, stdout=open(judge_log, "w"), stderr=subprocess.STDOUT
)

wait_for_server("http://127.0.0.1:8001", f"Judge ({JUDGE_REPO})", judge_log, judge_proc)

# ==============================================================================
# 6. Pre-Flight Check + Full 799-Case x 3-Pass Benchmark Execution
# ==============================================================================
harness_env = os.environ.copy()
harness_env["PYTHONPATH"] = f"{REPO_ROOT}:{REPO_ROOT / 'indicbankbench'}:" + harness_env.get("PYTHONPATH", "")

print("\n[6/8] Running 1-Case Pre-Flight Check...")
subprocess.run(
    [
        sys.executable, "-m", "harness.cli", "run",
        "--run-id", "preflight_check",
        "--passes", "1",
        "--cases", str(DATA_DIR / "case_bank/accounts_and_transactions/agentic/balance_gate.002.json"),
        "--fresh",
    ],
    cwd=str(REPO_ROOT),
    env=harness_env,
    check=True,
)

preflight_err = REPO_ROOT / "indicbankbench/results/preflight_check/cases/pass1/errors.jsonl"
if preflight_err.exists() and preflight_err.read_text().strip():
    print("[!] PRE-FLIGHT ERROR DETECTED:\n", preflight_err.read_text())
    raise RuntimeError("Pre-flight check failed.")

print(f"\n[6/8] Pre-flight E:0 verified! Launching Full 799-Case x 3-Pass Run (concurrency={concurrency})...")
subprocess.run(
    [
        sys.executable, "-m", "harness.cli", "run",
        "--run-id", RUN_ID,
        "--passes", "3",
        "--concurrency", str(concurrency),
        "--fresh",
        "--candidate-label", "Tiny-Aya-Fire-IndicBank (FT)",
    ],
    cwd=str(REPO_ROOT),
    env=harness_env,
    check=False,  # harness may exit non-zero on a few errored case-runs; the retry below fills them in
)

# Auto-retry once without --fresh to fill any transient timeouts
subprocess.run(
    [
        sys.executable, "-m", "harness.cli", "run",
        "--run-id", RUN_ID,
        "--passes", "3",
        "--concurrency", "4",
        "--candidate-label", "Tiny-Aya-Fire-IndicBank (FT)",
    ],
    cwd=str(REPO_ROOT),
    env=harness_env,
    check=False,
)

# ==============================================================================
# 7. Parse Results, Compare vs. Base Model & Paper Baselines, and Plot Figure 3
# ==============================================================================
RESULT_DIR = REPO_ROOT / "indicbankbench" / "results" / RUN_ID
summary_path = RESULT_DIR / "summary.json"
report_path = RESULT_DIR / "REPORT.md"

summary = json.loads(summary_path.read_text())
TOTAL_CASES = 799

ft_strict_n = int(summary.get("strict_pass", 0))
ft_pass3_n = int(summary.get("any_pass", 0))
ft_strict_pct = round((ft_strict_n / TOTAL_CASES) * 100.0, 1)
ft_pass3_pct = round((ft_pass3_n / TOTAL_CASES) * 100.0, 1)
ft_gap_pct = round(ft_pass3_pct - ft_strict_pct, 1)

print("\n" + "=" * 80)
print("OFFICIAL REPORT.MD FOR FINE-TUNED MODEL:")
print("=" * 80)
if report_path.exists():
    print(report_path.read_text())

PAPER_AND_BASE_MODELS = [
    {"model": "DeepSeek-V4-Pro-0813",            "strict_pass_n": 465, "strict_pass_pct": 58.2, "pass_at_3_n": 574, "pass_at_3_pct": 71.8, "gap_pct": 13.6},
    {"model": "Grok 4.6",                        "strict_pass_n": 464, "strict_pass_pct": 58.1, "pass_at_3_n": 561, "pass_at_3_pct": 70.2, "gap_pct": 12.1},
    {"model": "Qwen3.8-2.4T-A95B",               "strict_pass_n": 462, "strict_pass_pct": 57.8, "pass_at_3_n": 591, "pass_at_3_pct": 74.0, "gap_pct": 16.1},
    {"model": "Claude Opus 5",                   "strict_pass_n": 461, "strict_pass_pct": 57.7, "pass_at_3_n": 570, "pass_at_3_pct": 71.3, "gap_pct": 13.6},
    {"model": "GLM-5.3-Flash",                   "strict_pass_n": 454, "strict_pass_pct": 56.8, "pass_at_3_n": 594, "pass_at_3_pct": 74.3, "gap_pct": 17.5},
    {"model": "DeepSeek-V4-Flash-0731",          "strict_pass_n": 433, "strict_pass_pct": 54.2, "pass_at_3_n": 580, "pass_at_3_pct": 72.6, "gap_pct": 18.4},
    {"model": "Gemma 4 31B IT",                  "strict_pass_n": 432, "strict_pass_pct": 54.1, "pass_at_3_n": 525, "pass_at_3_pct": 65.7, "gap_pct": 11.6},
    {"model": "Gemini 3.8 Flash",                "strict_pass_n": 426, "strict_pass_pct": 53.3, "pass_at_3_n": 512, "pass_at_3_pct": 64.1, "gap_pct": 10.8},
    {"model": "MiniMax M3",                      "strict_pass_n": 385, "strict_pass_pct": 48.2, "pass_at_3_n": 556, "pass_at_3_pct": 69.6, "gap_pct": 21.4},
    {"model": "Gemma 4 26B A4B IT",              "strict_pass_n": 379, "strict_pass_pct": 47.4, "pass_at_3_n": 521, "pass_at_3_pct": 65.2, "gap_pct": 17.8},
    {"model": "Gemma 4 E4B IT",                  "strict_pass_n": 349, "strict_pass_pct": 43.7, "pass_at_3_n": 480, "pass_at_3_pct": 60.1, "gap_pct": 16.4},
    {"model": "Tiny-Aya-Fire (3.35B Base)",      "strict_pass_n": 24,  "strict_pass_pct": 3.0,  "pass_at_3_n": 134, "pass_at_3_pct": 16.8, "gap_pct": 13.8},
    {"model": "Tiny-Aya-Fire-IndicBank (FT)",    "strict_pass_n": ft_strict_n, "strict_pass_pct": ft_strict_pct, "pass_at_3_n": ft_pass3_n, "pass_at_3_pct": ft_pass3_pct, "gap_pct": ft_gap_pct},
]

df_leaderboard = pd.DataFrame(PAPER_AND_BASE_MODELS).sort_values(
    by=["strict_pass_pct", "pass_at_3_pct"], ascending=False
).reset_index(drop=True)
df_leaderboard.index = df_leaderboard.index + 1
df_leaderboard.index.name = "Rank"

csv_path = RESULT_DIR / "leaderboard_comparison.csv"
df_leaderboard.to_csv(csv_path)

print("\n=== FULL INDICBANKBENCH LEADERBOARD (WITH BASE vs. FINE-TUNED) ===")
print(df_leaderboard.to_string())

# Plot Figure 3 Dumbbell Comparison
df_plot = df_leaderboard.sort_values(by=["strict_pass_pct", "pass_at_3_pct"], ascending=True).reset_index(drop=True)

plt.rcParams["font.family"] = "serif"
fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
y_pos = np.arange(len(df_plot))

NAVY_DOT = "#1f3b4d"
GOLD_SQUARE = "#c88a2b"
LINE_COLOR = "#b0bec5"
BASE_COLOR = "#7f8c8d"
FT_COLOR = "#16a085"  # Emerald highlight for Fine-Tuned model

ax.xaxis.grid(True, linestyle="-", alpha=0.25, color="#cfd8dc", zorder=0)
ax.set_axisbelow(True)

for idx, row in df_plot.iterrows():
    is_ft = "IndicBank (FT)" in row["model"]
    is_base = "Base)" in row["model"]
    line_col = FT_COLOR if is_ft else (BASE_COLOR if is_base else LINE_COLOR)
    lw = 2.5 if is_ft else (1.8 if is_base else 1.2)

    ax.plot([row["strict_pass_pct"], row["pass_at_3_pct"]], [idx, idx], color=line_col, linewidth=lw, zorder=2)
    ax.scatter(
        row["strict_pass_pct"], idx,
        color=FT_COLOR if is_ft else (BASE_COLOR if is_base else NAVY_DOT),
        s=75 if is_ft else 50, marker="o", zorder=3
    )
    ax.scatter(
        row["pass_at_3_pct"], idx,
        color=GOLD_SQUARE,
        edgecolors=FT_COLOR if is_ft else "none",
        linewidths=1.8 if is_ft else 0,
        s=75 if is_ft else 55, marker="s", zorder=3
    )

    gap_sign = "+" if row["gap_pct"] >= 0 else ""
    ax.text(
        row["pass_at_3_pct"] + 1.0, idx, f"{gap_sign}{row['gap_pct']:.1f}",
        va="center", ha="left", fontsize=9.5,
        fontweight="bold" if is_ft else "normal",
        color=FT_COLOR if is_ft else "#263238"
    )

ax.set_yticks(y_pos)
yticklabels = ax.set_yticklabels(df_plot["model"], fontsize=10.5)
for label in yticklabels:
    if "IndicBank (FT)" in label.get_text():
        label.set_color(FT_COLOR)
        label.set_fontweight("bold")
    elif "Base)" in label.get_text():
        label.set_color("#c0392b")

min_score = min(df_plot["strict_pass_pct"].min(), 40.0)
x_min = max(0.0, np.floor((min_score - 5.0) / 5.0) * 5.0) if min_score < 40.0 else 40.0
ax.set_xlim(x_min, 82)
ax.set_xlabel("Cases passed (%) across 799 cases", fontsize=11, labelpad=8)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)

strict_handle = mlines.Line2D([], [], color=NAVY_DOT, marker="o", linestyle="None", markersize=7, label="Strict pass³")
pass3_handle = mlines.Line2D([], [], color=GOLD_SQUARE, marker="s", linestyle="None", markersize=7, label="At least once (pass@3)")
ax.legend(handles=[strict_handle, pass3_handle], loc="upper center", bbox_to_anchor=(0.48, 1.12), ncol=2, frameon=False, fontsize=10.5)

plt.tight_layout()
plot_path = RESULT_DIR / "indicbankbench_finetuned_comparison.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"[7/8] Saved Figure 3 comparison plot to {plot_path}")

# ==============================================================================
# 8. Upload All Evaluation Artifacts to Your Hugging Face Model Repo
# ==============================================================================
print(f"[8/8] Uploading evaluation results & plot to {CANDIDATE_REPO}/tree/main/eval_results ...")
for artifact in [plot_path, report_path, summary_path, csv_path]:
    if artifact.exists():
        api.upload_file(
            path_or_fileobj=str(artifact),
            path_in_repo=f"eval_results/{artifact.name}",
            repo_id=CANDIDATE_REPO,
            repo_type="model",
            commit_message=f"Add IndicBankBench evaluation artifact: {artifact.name}",
        )

print("\n" + "=" * 80)
print(f"SUCCESS! View your Figure 3 Plot & Full Report at:")
print(f"https://huggingface.co/{CANDIDATE_REPO}/tree/main/eval_results")
print("=" * 80)

candidate_proc.terminate()
judge_proc.terminate()