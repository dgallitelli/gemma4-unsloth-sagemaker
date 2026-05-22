"""Local SageMaker training-job orchestrator (no compute here).

Pulls HF_TOKEN from AWS Secrets Manager, launches a single Training job that
downloads the dataset, formats it for Gemma 4, and fine-tunes with QLoRA + Unsloth.

Aligned on SageMaker Python SDK v3 throughout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.training.configs import (
    Compute,
    OutputDataConfig,
    SourceCode,
    StoppingCondition,
)
from sagemaker.train import ModelTrainer

HERE = Path(__file__).resolve().parent

# us-east-1 SageMaker Training on-demand hourly rates (USD), verified 2026-05
# via `aws pricing get-products`. Sanity-check only; not authoritative for billing.
# Inference (endpoint hosting) rates differ — these are training rates only.
INSTANCE_HOURLY_USD = {
    "ml.g6e.xlarge": 1.86,
    "ml.g6e.2xlarge": 2.24,
    "ml.g7e.2xlarge": 3.36,
    "ml.g6e.12xlarge": 10.49,
}

# AWS HuggingFace training DLC. The SDK's image_uris.retrieve config lags
# behind ECR; the latest transformers 5.x triple is not registered as of
# May 2026, so we hardcode the ECR URI. Verify with:
#   aws ecr describe-images --registry-id 763104351884 \
#     --repository-name huggingface-pytorch-training --region <region>
HF_DLC_ACCOUNT = "763104351884"
HF_DLC_TAG = "2.9.0-transformers5.3.0-gpu-py312-cu130-ubuntu22.04"


def hf_training_image_uri(region: str) -> str:
    return (
        f"{HF_DLC_ACCOUNT}.dkr.ecr.{region}.amazonaws.com/"
        f"huggingface-pytorch-training:{HF_DLC_TAG}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--secret-name", default="huggingface/token")
    p.add_argument("--secret-key", default="HF_TOKEN")
    p.add_argument("--dataset", default="mlabonne/guanaco-llama2-1k")
    p.add_argument("--instance-type", default="ml.g6e.xlarge")
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--max-run-seconds", type=int, default=2 * 3600)
    p.add_argument("--volume-size-gb", type=int, default=200)
    p.add_argument(
        "--keep-alive-seconds",
        type=int,
        default=3600,
        help="Warm-pool keep-alive after job completion (max 3600). Silently no-ops "
        "if your account's warm-pool quota for this instance type is 0.",
    )
    p.add_argument("--bucket-prefix", default="gemma4-poc")
    p.add_argument(
        "--role-arn",
        default=None,
        help="SageMaker execution role ARN. Required outside Studio; "
        "v3 get_execution_role() returns the caller identity, which SageMaker can't assume.",
    )
    p.add_argument(
        "--merge",
        action="store_true",
        help="Train, then merge adapter into a 4-bit checkpoint inside the same "
        "training job. Requires ~96 GB GPU; auto-bumps --instance-type to "
        "ml.g7e.2xlarge if the user-supplied default is still in effect. "
        "Output tarball contains adapter/ and merged_4bit/ side by side.",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def get_hf_token(region: str, secret_name: str, secret_key: str) -> str:
    client = boto3.client("secretsmanager", region_name=region)
    try:
        resp = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code == "ResourceNotFoundException":
            print(
                f"Secret '{secret_name}' not found in {region}. Create it with:\n"
                f"  aws secretsmanager create-secret --name {secret_name} \\\n"
                f"    --secret-string '{{\"{secret_key}\":\"hf_xxx\"}}' --region {region}",
                file=sys.stderr,
            )
        elif code == "AccessDeniedException":
            print(
                f"Access denied reading '{secret_name}'. Your local AWS credentials need "
                f"secretsmanager:GetSecretValue on this secret's ARN.",
                file=sys.stderr,
            )
        raise
    secret_str = resp.get("SecretString")
    if not secret_str:
        raise RuntimeError(f"Secret '{secret_name}' has no SecretString (binary secret?).")
    payload = json.loads(secret_str)
    if secret_key not in payload:
        raise RuntimeError(
            f"Secret '{secret_name}' missing key '{secret_key}'. Found keys: {list(payload)}"
        )
    return payload[secret_key]


def estimate_cost_usd(billable_seconds: int, instance_type: str) -> float:
    rate = INSTANCE_HOURLY_USD.get(instance_type)
    if rate is None:
        return 0.0
    return rate * billable_seconds / 3600.0


def describe_training_job(trainer: ModelTrainer, region: str) -> dict:
    """Fetch the underlying training-job description via boto3 (v3 attribute access varies)."""
    name = (
        getattr(trainer, "latest_training_job_name", None)
        or getattr(getattr(trainer, "_latest_training_job", None), "name", None)
    )
    if not name:
        return {}
    sm = boto3.client("sagemaker", region_name=region)
    return sm.describe_training_job(TrainingJobName=name)


def main() -> int:
    args = parse_args()

    # Merging dequantizes the base layer-by-layer to bf16 before re-quantizing to
    # NF4. Peak VRAM during the bf16 hop is ~1 layer × bf16 + the rest still in
    # NF4 — fits in 48 GB in theory, but Unsloth's "merged_4bit_forced" path is
    # only routinely tested on ≥80 GB. Bump silently only if the user kept the
    # default; otherwise trust them.
    if args.merge and args.instance_type == "ml.g6e.xlarge":
        args.instance_type = "ml.g7e.2xlarge"
        print(
            "--merge requires more headroom than ml.g6e.xlarge offers; "
            "auto-bumping instance to ml.g7e.2xlarge (96 GB).",
            flush=True,
        )

    print(f"Region: {args.region} | instance: {args.instance_type}", flush=True)

    print(f"Fetching HF_TOKEN from secret '{args.secret_name}' ({args.region})", flush=True)
    hf_token = get_hf_token(args.region, args.secret_name, args.secret_key)

    if args.dry_run:
        print("Dry run — exiting before any SageMaker API calls.", flush=True)
        return 0

    boto_session = boto3.Session(region_name=args.region)
    session = Session(boto_session=boto_session)
    if args.role_arn:
        role = args.role_arn
    else:
        role = get_execution_role(session)
        if ":role/aws-reserved/sso." in role or ":assumed-role/" in role:
            print(
                f"\nERROR: get_execution_role() returned the caller identity ({role}).\n"
                f"SageMaker won't be able to assume this role. Pass --role-arn explicitly:\n"
                f"  --role-arn arn:aws:iam::<account>:role/service-role/AmazonSageMaker-ExecutionRole-...\n"
                f"List candidates with:\n"
                f"  aws iam list-roles --query \"Roles[?contains(RoleName,'SageMaker')].Arn\"",
                file=sys.stderr,
            )
            return 2
    bucket = session.default_bucket()
    print(f"Role: {role} | bucket: s3://{bucket}", flush=True)

    training_image = hf_training_image_uri(args.region)
    print(f"Training image: {training_image}", flush=True)

    hyperparameters: dict[str, Any] = {
        "model_id": "google/gemma-4-31B-it",
        "dataset": args.dataset,
        "max_seq_length": args.max_seq_length,
        "lora_rank": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 1,
        "learning_rate": 2e-4,
        "warmup_steps": 10,
        "merge": str(args.merge).lower(),
    }

    trainer = ModelTrainer(
        training_image=training_image,
        source_code=SourceCode(
            source_dir=str(HERE / "src"),
            entry_script="train.py",
            requirements="requirements.txt",
        ),
        compute=Compute(
            instance_type=args.instance_type,
            instance_count=1,
            volume_size_in_gb=args.volume_size_gb,
            keep_alive_period_in_seconds=args.keep_alive_seconds,
        ),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=args.max_run_seconds),
        hyperparameters=hyperparameters,
        environment={"HF_TOKEN": hf_token, "TRANSFORMERS_CACHE": "/tmp/hf_cache"},
        output_data_config=OutputDataConfig(
            s3_output_path=f"s3://{bucket}/{args.bucket_prefix}/training/"
        ),
        role=role,
        sagemaker_session=session,
        base_job_name="gemma4-unsloth-qlora",
    )

    started = time.time()
    try:
        trainer.train(wait=True, logs=True)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        msg = e.response.get("Error", {}).get("Message", "")
        if "ResourceLimit" in code or "Capacity" in msg or "capacity" in msg.lower():
            print(
                f"\nCapacity/limit error on {args.instance_type}: {msg}\n"
                f"Try: --instance-type ml.g6e.2xlarge, or change --region.",
                file=sys.stderr,
            )
        raise

    wall_seconds = int(time.time() - started)

    train_desc = describe_training_job(trainer, args.region)
    artifacts = train_desc.get("ModelArtifacts", {}).get("S3ModelArtifacts")
    if artifacts:
        print(f"Adapter artifacts: {artifacts}", flush=True)

    # Wall-time fallback over-reports cost for jobs that Stopped early — uploads
    # and instance startup don't bill at the full rate. Acceptable for a sanity
    # check; treat the printed number as an upper bound.
    train_billable = int(train_desc.get("BillableTimeInSeconds") or 0) or wall_seconds
    train_cost = estimate_cost_usd(train_billable, args.instance_type)
    print(
        f"\nEstimated cost: ${train_cost:.2f} ({args.instance_type} × "
        f"{train_billable / 60:.1f} min)",
        flush=True,
    )
    if train_cost > 10.0:
        print("WARNING: estimated training cost > $10 — review job duration.", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
