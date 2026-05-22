# Gemma 4 31B QLoRA Fine-Tuning on Amazon SageMaker AI

A reference implementation for fine-tuning [Google Gemma 4 31B](https://huggingface.co/google/gemma-4-31B-it) with **QLoRA + [Unsloth](https://unsloth.ai/)** on Amazon SageMaker AI, fitting on a **single L40S GPU (48 GB)**.

The repo is a thin orchestration layer: a local script (or notebook) launches a single managed SageMaker Training job. There is **no local GPU compute** — all heavy lifting runs on managed SageMaker infrastructure.

## Why this exists

Gemma 4 has several quirks (Flash Attention incompatibility on global-attention layers, custom linear modules that PEFT doesn't auto-detect, a required data-collator field, a `transformers` version floor) that make naive fine-tuning fail. Unsloth patches all of these. This repo packages those patches into a SageMaker workflow that runs on the cheapest GPU instance that fits a 31B 4-bit model.

## Architecture

```
┌──────────────────────────────┐
│ launch.py / launch.ipynb     │  local orchestrator (no compute)
└──────────────┬───────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ SageMaker Training Job                  │
│ ml.g6e.xlarge (1× L40S, 48 GB)          │
│                                         │
│ src/train.py                            │
│   ├─ download dataset from HF Hub       │
│   ├─ re-template Llama 2 → Gemma 4      │
│   ├─ load Unsloth FastModel (4-bit)     │
│   ├─ token-length filter                │
│   ├─ LoRA + SFTTrainer                  │
│   └─ save adapter → /opt/ml/model/      │
└────────────────┬────────────────────────┘
                 ▼
            S3 (LoRA adapter)
```

The dataset is small (1K rows for the default `mlabonne/guanaco-llama2-1k`); download + format + filter all run inside the training container at startup. No separate Processing job.

## Repository layout

```
launch.py                     # local CLI orchestrator
launch.ipynb                  # notebook variant (SageMaker Studio / Jupyter)
src/
  train.py                    # runs in Training container (GPU)
  requirements.txt            # installed at training-container startup
```

## Prerequisites

- AWS account with permission to launch SageMaker Training Jobs in a region with `g6e` (or `g7e`, for `--merge`) capacity. `us-east-1` and `us-west-2` are both viable; check Service Quotas for `ml.g6e.xlarge` before committing — and additionally `ml.g7e.2xlarge` if you intend to run with `--merge`.
- A **SageMaker execution role** (e.g. one with `AmazonSageMakerFullAccess`) whose trust policy allows `sagemaker.amazonaws.com` to assume it. Pass its ARN to `launch.py` via `--role-arn`. *Note:* `get_execution_role()` returns the caller identity outside SageMaker Studio — for SSO callers this is an `aws-reserved` role that SageMaker can't assume, so `--role-arn` is mandatory for non-Studio runs.
- Your **local** AWS credentials (the ones running `launch.py`) need:
  - `secretsmanager:GetSecretValue` on the HF token secret
  - `sagemaker:CreateTrainingJob` / `DescribeTrainingJob`
  - `iam:PassRole` for the execution role
  - `s3:*` on the SageMaker default bucket
- A [Hugging Face access token](https://huggingface.co/settings/tokens) stored in **AWS Secrets Manager** (default secret name: `huggingface/token`, JSON shape `{"HF_TOKEN": "hf_..."}`). The launcher fetches it at runtime and injects it into the training job's `environment` — the token is never committed and never typed into a notebook cell.
- Python 3.10+ locally for `launch.py` with `sagemaker>=3.12` (the v3 SDK with `ModelTrainer`) and `boto3`.

## Quickstart

```bash
# 1. Configure AWS credentials (any standard method)
aws configure  # or assume-role / SSO / env vars

# 2. Install the SageMaker SDK v3 in a virtualenv
python3 -m venv .venv && source .venv/bin/activate
pip install 'sagemaker>=3.12' boto3

# 3. Store your HF token in Secrets Manager (one-time, per region)
aws secretsmanager create-secret \
  --name huggingface/token \
  --secret-string '{"HF_TOKEN":"hf_xxx"}' \
  --region <your-region>

# 4. Find a SageMaker execution role
aws iam list-roles --query "Roles[?contains(RoleName,'SageMaker-ExecutionRole')].Arn"

# 5. Run the pipeline (from the repo root — paths are resolved relative to launch.py)
python launch.py \
  --region <your-region> \
  --role-arn arn:aws:iam::<account>:role/service-role/AmazonSageMaker-ExecutionRole-<id>
```

### Train and merge in one job (recommended for serving)

Add `--merge` to produce a self-contained 4-bit checkpoint that vLLM/LMI can serve directly — no LoRA loader required at inference time:

```bash
python launch.py \
  --region <your-region> \
  --role-arn <arn> \
  --merge
```

When `--merge` is set and the instance type is left at the default, the launcher auto-bumps to `ml.g7e.2xlarge` (96 GB) — Unsloth's layer-by-layer merge-and-requantize path needs the headroom. The output tarball contains both `adapter/` (raw LoRA, useful for future redeployment) and `merged_4bit/` (NF4 checkpoint ready for `option.quantization=bitsandbytes`).

The launcher creates the training job, streams CloudWatch logs, and prints the final adapter S3 URI plus a cost estimate.

For a serving-ready 4-bit checkpoint instead of just the LoRA adapter, append `--merge` — see [the next section](#train-and-merge-in-one-job-recommended-for-serving).

To run from a notebook instead, open `launch.ipynb`.

## Cost & runtime

| Stage | Instance | Hourly | Typical duration | Cost |
|---|---|---|---|---|
| Container startup (Unsloth git install + model download) | `ml.g6e.xlarge` | ~$1.86 | 10–15 min | ~$0.30 |
| Training (1K samples, 1 epoch) | `ml.g6e.xlarge` | ~$1.86 | 20–30 min | ~$0.60 |
| **Full POC run** | | | ~30–45 min | **~$1–2** |

Costs are illustrative and depend on region and your AWS pricing.

## Loading the trained adapter

The training job writes `model.tar.gz` to S3. Layout depends on whether you ran with `--merge`:

| Run | Tarball contents |
|---|---|
| Default | `adapter_config.json`, `adapter_model.safetensors` at the archive root |
| `--merge` | `adapter/` (raw LoRA, same files as above) **and** `merged_4bit/` (self-contained NF4 checkpoint) |

```python
from unsloth import FastModel
from peft import PeftModel

# Default run — adapter files at the extraction root
base, tokenizer = FastModel.from_pretrained("google/gemma-4-31B-it", load_in_4bit=True)
model = PeftModel.from_pretrained(base, "/path/to/extracted/")

# After --merge — load the merged 4-bit checkpoint directly (no PEFT needed)
model, tokenizer = FastModel.from_pretrained(
    "/path/to/extracted/merged_4bit/", load_in_4bit=True
)
```

For SageMaker / vLLM / DJL Serving deployment, prefer `--merge` — the `merged_4bit/` directory is a self-contained NF4 checkpoint that vLLM and LMI can serve directly with `option.quantization=bitsandbytes`, no adapter loader required at inference time.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Could not assume role aws-reserved/sso...` | Your caller is SSO; `get_execution_role()` returned the wrong ARN | Pass `--role-arn` explicitly |
| `Secret 'huggingface/token' not found` in region X | Secret only exists in another region | Create per-region or pass `--secret-name` to point at an existing one |
| OOM during training | Activation memory tight on 1× L40S | Switch to `ml.g6e.2xlarge`, or reduce `--max-seq-length` |
| `Gemma4Config` not registered | DLC's shipped `transformers` is too old | Confirm `transformers==5.5.0` in `src/requirements.txt` and that the training-job logs show pip installing it. The pin is exact: `unsloth_zoo` caps `transformers <= 5.5.0` and Gemma 4 needs `>= 5.5.0`, leaving exactly one version that satisfies both. |
| Flash Attention crash with `head_dim=512` | Bypassed Unsloth's patches | Ensure `unsloth` imports before `transformers` in `train.py` |
| Training job stuck in `Starting` for 5–10 min | Unsloth git install + model download | Expected; monitor CloudWatch |
| Capacity errors on `g6e.xlarge` | Region capacity | Try `--instance-type ml.g6e.2xlarge` or change `--region` |

## Design notes

- **No SageMaker Training Compiler** — incompatible with Gemma 4.
- **Script Mode (BYOS)** — no JumpStart fine-tuning recipe exists for Gemma 4.
- **Hugging Face DLC** — used as the base container (`huggingface-pytorch-training:2.9.0-transformers5.3.0-gpu-py312-cu130-ubuntu22.04`); `requirements.txt` overrides the shipped `transformers` version.
- **`volume_size_in_gb=200`** — covers the 4-bit weights (~35 GB), dataset, and intermediate checkpoints.
- **SDK v3 throughout** — `sagemaker.train.ModelTrainer` + `sagemaker.core.helper.session_helper`; no v2 idioms.

## Not recommended

- `p4d` / `p5` / `p5e` — overkill for a 31B QLoRA POC.
- `ml.g6e.12xlarge` — 4× L40S; needed only if you can't get any single-GPU instance and don't mind the cost.

## License

Apache 2.0 — see [LICENSE](LICENSE). Gemma 4 is distributed under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms); review and accept before downloading the model.

## Acknowledgements

- [Google Gemma](https://ai.google.dev/gemma) for the model weights.
- [Unsloth](https://github.com/unslothai/unsloth) for the Gemma 4 patches and 2× speed-up.
- The Hugging Face [TRL](https://github.com/huggingface/trl) / [PEFT](https://github.com/huggingface/peft) / [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) ecosystem.
- [AWS Deep Learning Containers](https://github.com/aws/deep-learning-containers) for the prebuilt training image.
