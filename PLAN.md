# Implementation Plan

Working plan for the Gemma 4 31B + QLoRA + Unsloth POC on SageMaker. Authoritative spec is `CLAUDE.md`; user-facing docs are `README.md`.

## Architecture (final)

**Single SageMaker Training job.** No Processing job. The dataset is small (1K rows), so download + format + length-filter all run inside the training container at startup. Originally split into Processing + Training, but the Processing job was doing trivial work that didn't justify a separate image, an extra IAM hop, or two failure surfaces.

```
launch.py (local)
  └─ ModelTrainer.train()  ─→ ml.g6e.xlarge (1× L40S 48GB)
                              src/train.py
                                ├─ download dataset from HF Hub
                                ├─ regex-parse Llama 2 [INST] markers
                                ├─ render Gemma 4 chat tokens
                                ├─ load Unsloth FastModel (4-bit)
                                ├─ token-length filter
                                ├─ LoRA + SFTTrainer
                                └─ save adapter → /opt/ml/model/ → S3
```

## Definition of done

| # | Criterion | Concrete signal |
|---|---|---|
| 1 | Training job runs end-to-end | `DescribeTrainingJob` `TrainingJobStatus == "Completed"` |
| 2 | Loss decreases | `train.py` asserts `loss[late] < 0.9 × loss[early]` and exits non-zero on failure |
| 3 | Adapter artifacts present | `adapter_model.safetensors` + `adapter_config.json` in `model.tar.gz` (printed by `train.py` via `os.listdir(output_dir)` before exit) |
| 4 | Peak VRAM < 48 GB | `train.py` logs `torch.cuda.max_memory_allocated() / 1e9` post-load and post-training; asserts < 45 GB after load |
| 5 | No Flash Attention | `train.py` asserts `model.config._attn_implementation != "flash_attention_2"` post-load and logs the actual value |
| 6 | Total cost < $10 | `launch.py` prints `BillableTimeInSeconds × hourly_rate` at end; warns if > $10 |

## SageMaker SDK choice — v3 only

Aligned on **SageMaker Python SDK v3** per the user's global rule.

Verified API surface (research 2026-05-21, `sagemaker==3.12.0`):

```python
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import Compute, OutputDataConfig, SourceCode, StoppingCondition
from sagemaker.core.helper.session_helper import Session, get_execution_role
```

Key v3 gotchas (vs. v2 muscle memory):
- `ModelTrainer` has **no** `volume_size_in_gb=` or `max_runtime_in_seconds=` kwargs — they nest in `Compute(volume_size_in_gb=...)` and `StoppingCondition(max_runtime_in_seconds=...)`.
- Training is started with `.train()`, **not** `.fit()`.
- `output_data_config=OutputDataConfig(s3_output_path=...)`, **not** `output_path=`.
- `logs=True` (boolean), not `'All'`.
- **`get_execution_role()` outside Studio falls back to caller identity.** For SSO callers this returns `arn:.../aws-reserved/sso.amazonaws.com/...`, which SageMaker can't assume (`sagemaker.amazonaws.com` is not a trusted principal). `launch.py` detects this and refuses to proceed unless `--role-arn` is passed explicitly.
- **`ProcessingJob` has no `.describe()` method in v3.** The training-job description is fetched via `boto3.client("sagemaker").describe_training_job(TrainingJobName=...)` instead.

## File-level plan

### `src/train.py`

Runs in the SageMaker Training container (`ml.g6e.xlarge`, GPU). Self-contained: data download + format + filter + train + save.

- **Imports**: `from unsloth import FastModel` BEFORE `transformers` / `trl` / `peft`. Unsloth's monkey-patches register on import; out-of-order imports silently disable the Gemma 4 fixes.
- **Argparse**: every hyperparameter from `CLAUDE.md` plus `--model_id`, `--dataset`, `--max_drop_pct`, `--output_dir` (default `os.environ.get("SM_MODEL_DIR", "/opt/ml/model")`), `--max_steps` (default `-1` — used for the load-only smoke test).
- **HF auth**: `HF_TOKEN` arrives via the trainer's `environment=` dict; `huggingface_hub.login()` early.
- **Dataset prep**: `load_and_format_dataset()` regex-parses Llama 2 `[INST]...[/INST]` markers and re-templates with hand-rolled Gemma 4 chat tokens (`<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n{assistant}<end_of_turn>\n`). System messages fold into the first user turn (Gemma rejects the `system` role). `max_drop_pct` enforces a parse-failure budget.
- **Model load**: `FastModel.from_pretrained(args.model_id, load_in_4bit=True, max_seq_length=args.max_seq_length)` — positional `model_id` to dodge upstream kwarg renames. Unsloth handles `attn_implementation="sdpa"`, `Gemma4ClippableLinear`, and `mm_token_type_ids`.
- **Post-load asserts** (DoD #4, #5):
  ```python
  attn_impl = model.config._attn_implementation
  if attn_impl == "flash_attention_2": return 3
  vram_gb = torch.cuda.max_memory_allocated() / 1e9
  if vram_gb >= 45.0: return 3
  ```
- **Length filter**: token-level filter against the loaded Gemma 4 tokenizer (not earlier — the Gemma 4 tokenizer is heavy and the upstream sklearn DLC was Py3.9, which can't host it).
- **LoRA**: `FastModel.get_peft_model(model, r=..., target_modules=GEMMA4_LORA_TARGETS, use_gradient_checkpointing="unsloth")`. Explicit module list (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`); `"all-linear"` doesn't always pass through Unsloth's wrapper.
- **Trainer (TRL ≥ 0.12 API)**: `dataset_text_field="text"` lives on `SFTConfig`, not `SFTTrainer.__init__`. `processing_class=tokenizer` (renamed from `tokenizer=` in TRL 0.12).
- **Loss-decrease assertion (DoD #2)**: aggregate first-5 vs. last-5 loss values from `trainer.state.log_history`; require ≥10% drop. Skipped if fewer than 20 logged steps.
- **Save**: `trainer.model.save_pretrained(args.output_dir)` for the default (adapter-only) path. With `--merge`, additionally call `model.save_pretrained_merged(<output_dir>/merged_4bit, tokenizer, save_method="merged_4bit_forced")` to produce a self-contained NF4 checkpoint suitable for direct vLLM/LMI serving (sidesteps known vLLM Gemma 4 + LoRA bugs that affect runtime adapter loading).

### `src/requirements.txt`

```
unsloth @ git+https://github.com/unslothai/unsloth.git
unsloth_zoo>=2026.4.6
transformers==5.5.0
```

The DLC pre-installs `trl`, `peft`, `bitsandbytes`, `datasets`, `accelerate`, and `huggingface_hub`; we only override `transformers` to the exact version that satisfies both the Gemma 4 architecture floor and `unsloth_zoo`'s upper bound.

Why no extras tag on Unsloth: docs don't quote a `cu130-torch290` token; the unpinned-extras install picks the matched wheel automatically based on the runtime CUDA + Torch (the DLC ships cu130 + Torch 2.9). If pip resolves a slow source build, fall back to `unsloth[cu128-torch280] @ git+...`. Document the resolved combo in the commit message after first success.

### `launch.py`

Local CLI orchestrator — no compute. Single-job pipeline.

- **Argparse**: `--region` (default `us-west-2`), `--secret-name`/`--secret-key`, `--dataset`, `--instance-type` (default `ml.g6e.xlarge`), `--max-seq-length`, `--max-run-seconds` (default `2*3600` — tighter than the spec's 6h to fail fast), `--volume-size-gb` (200), `--bucket-prefix`, `--role-arn`, `--dry-run`.
- **Secrets**: `boto3.client("secretsmanager")` → `get_secret_value` → `json.loads(...)[secret_key]`. Surfaces `ResourceNotFoundException` / `AccessDeniedException` with explicit remediation.
- **Role**: explicit `--role-arn` strongly preferred. If `get_execution_role()` returns an SSO-reserved or assumed-role ARN, refuse to proceed and print how to find a real SageMaker execution role.
- **Training image**: hardcoded ECR URI (`763104351884.dkr.ecr.<region>.amazonaws.com/huggingface-pytorch-training:2.9.0-transformers5.3.0-gpu-py312-cu130-ubuntu22.04`). The SDK's `image_uris.retrieve` config doesn't know transformers 5.x as of May 2026.
- **`ModelTrainer`**: see code in `launch.py` — `Compute(volume_size_in_gb=..)`, `StoppingCondition(max_runtime_in_seconds=..)`, `OutputDataConfig(s3_output_path=..)`, hyperparameters dict with no secrets, `environment={"HF_TOKEN": ...}`.
- **Capacity error path**: catch `ClientError` with `ResourceLimitExceeded` / `Capacity*` and print "Try `--instance-type ml.g6e.2xlarge` or change region". Don't auto-retry.
- **Cost summary**: read `BillableTimeInSeconds` from `boto3 describe_training_job` (not from the v3 trainer object, which doesn't expose it). Multiply by a region-keyed rate map.

### `launch.ipynb`

Mirror of `launch.py` as notebook cells, for SageMaker Studio / Jupyter users. Same Secrets Manager call, same hyperparameters, same SDK v3 surface.

### Repo polish

- `.gitignore`: `.venv/`, `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, `.DS_Store`, `model/`, `*.tar.gz`.
- Header comment in `src/train.py` indicating where it runs and the import-order constraint.
- No `HF_TOKEN`, AWS account IDs, or specific bucket names hardcoded in source.

## Hyperparameter & secret flow

| Value | Source | Mechanism | Visible to job? |
|---|---|---|---|
| All training hyperparameters + `dataset` | `launch.py` `hyperparameters={}` | Delivered as `--key value` CLI args to `train.py` | Yes — visible in CloudWatch and `DescribeTrainingJob` |
| `HF_TOKEN` | Secrets Manager (fetched locally) | Passed via `environment={}` dict | Yes — but env vars are not echoed to CloudWatch by default |

## IAM

The **launcher's local AWS credentials** need:
- `secretsmanager:GetSecretValue` on the HF token secret
- `sagemaker:CreateTrainingJob`, `sagemaker:DescribeTrainingJob`
- `iam:PassRole` for the SageMaker execution role
- `s3:*` on the SageMaker default bucket

The **SageMaker execution role** needs:
- `s3:GetObject`, `s3:PutObject` on the default bucket
- `ecr:GetAuthorizationToken`, `ecr:BatchGetImage` on the DLC ECR registry
- *Not* `secretsmanager:GetSecretValue` — the launcher injects the token via `environment=`

## Lessons from execution (2026-05-21)

Recorded so the next maintainer doesn't re-discover them:

1. **`get_execution_role()` v3 is hostile to SSO callers.** Outside Studio, it returns the caller's identity ARN, which SageMaker can't assume. → `--role-arn` is now mandatory for non-Studio runs.
2. **`ProcessingOutput` v3 schema breaks v2 muscle memory.** It's a Pydantic model with `extra="forbid"`; S3 config nests inside `ProcessingS3Output(s3_uri=, local_path=, s3_upload_mode=)` instead of flat `source=`/`destination=` kwargs. Fixed before consolidation removed the Processing job entirely.
3. **`ProcessingJob` v3 has no `.describe()`.** Reach for `boto3.client("sagemaker").describe_*` instead.
4. **The sklearn DLC ships Python 3.9 + a pyarrow-vs-numpy ABI mismatch** (pyarrow built against numpy 2.x umath, but conda env has numpy 1.x). pip-installing `numpy<2` doesn't help — the loaded pyarrow imports `numpy._core.umath` regardless. Don't fight it; use the HuggingFace DLC for processing too, or skip the Processing job entirely (we did the latter).
5. **The Gemma 4 chat template can be hand-rolled.** `<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n{assistant}<end_of_turn>\n` — no tokenizer needed for formatting, only for length filtering.
6. **transformers 5.x requires Py 3.10+.** Anything Gemma 4 needs the GPU DLC's Python 3.12.

## Resolved findings

| Item | Resolved value | Confidence |
|---|---|---|
| HF training DLC | `2.9.0-transformers5.3.0-gpu-py312-cu130-ubuntu22.04` | High; verified by container pull |
| Unsloth extras | Unpinned, let installer auto-resolve | Medium — confirm at first successful run |
| Model ID | `google/gemma-4-31B-it` (uppercase `B`) | High |
| Dataset markers | `<s>[INST] ... [/INST] ... </s>` (Llama 2) | High |
| `transformers` pin | `==5.5.0` exact (only version satisfying Gemma 4 floor + `unsloth_zoo` upper bound) | High |
| SDK v3 latest | `sagemaker==3.12.0` | High |

## Remaining open items

Resolvable only at runtime:

1. **`g6e.xlarge` capacity in target region** — `launch.py` surfaces a clear error.
2. **`bitsandbytes` cu130 wheel availability** — confirm at install. If missing, downgrade DLC one generation.
3. **Unsloth wheel resolution** — confirm at install.
4. **Gemma 4 multimodal class quirks** — `FastModel.from_pretrained` should abstract; verified by post-load attention-impl + VRAM checks.

## Risks

- **VRAM**: ~18 GB Unsloth claim is for weights only. Activations at `seq_len=2048` × `batch=2` may push higher. Mitigation: load-only smoke test before full epoch (set `--max-steps 5` via hyperparameters).
- **Container startup**: Unsloth git install is 5–10 min — visible as "Starting" status with no progress. Mitigation: README troubleshooting; `--max-run-seconds` capped at 2h.
- **DLC drift**: HF DLC versions move. Mitigation: ECR URI is a single constant in `launch.py`; bump deliberately.

## Out of scope

- Multi-GPU / multi-node training.
- Model serving / endpoint deployment (the `--merge` flag produces a checkpoint that is *ready* to deploy on a SageMaker endpoint via DJL Serving / LMI, but the deploy script itself is out of scope for this POC).
- Bedrock import of the trained adapter.
- A separate Processing job (consolidated into `train.py` after first run revealed it added cost without value for this dataset size).
