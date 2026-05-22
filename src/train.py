"""Runs in SageMaker Training container (GPU). Import order: unsloth FIRST.

Downloads the instruction dataset, re-templates rows from Llama 2 markers
into Gemma 4 chat format, loads Gemma 4 31B in 4-bit via Unsloth, attaches
a LoRA adapter, fine-tunes with TRL's SFTTrainer, and saves the adapter to
/opt/ml/model/.

Self-contained — no separate Processing job. The dataset is small enough
(1K rows) that prep takes seconds inside the training container.
"""

# ruff: noqa: E402  — import order is load-bearing; do not let a formatter "fix" it.
from __future__ import annotations

# Unsloth's monkey-patches must register before transformers/trl/peft imports.
from unsloth import FastModel

import argparse
import os
import re
import sys

import torch
from datasets import Dataset, load_dataset
from huggingface_hub import login as hf_login
from trl import SFTConfig, SFTTrainer

# Llama 2 instruction template:
#   <s>[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n{user} [/INST] {assistant} </s>
# or simpler: <s>[INST] {user} [/INST] {assistant} </s>
LLAMA2_TURN_RE = re.compile(
    r"\[INST\]\s*(?:<<SYS>>\s*(?P<sys>.*?)\s*<</SYS>>\s*)?(?P<user>.*?)\s*\[/INST\]\s*(?P<assistant>.*?)\s*(?=<s>|\[INST\]|$)",
    re.DOTALL,
)

# Gemma 4 chat format. Bos/eos handled by the trainer's tokenizer.
GEMMA4_TURN = (
    "<start_of_turn>user\n{user}<end_of_turn>\n"
    "<start_of_turn>model\n{assistant}<end_of_turn>\n"
)

# Gemma 4 31B linear layer names — explicit list. "all-linear" is a peft-config
# convention that Unsloth's wrapper doesn't always pass through; hand-listing
# the modules guarantees LoRA actually attaches.
GEMMA4_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def _bool_arg(s: str) -> bool:
    """Parse a string into bool. SageMaker hyperparameters always serialize as
    `--key value`, so `action="store_true"` doesn't survive the round-trip."""
    return s.strip().lower() in ("1", "true", "yes", "y")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="google/gemma-4-31B-it")
    p.add_argument("--dataset", default="mlabonne/guanaco-llama2-1k")
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--max_drop_pct", type=float, default=5.0)
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=-1)  # -1 = use num_train_epochs
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--output_dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    p.add_argument(
        "--merge",
        type=_bool_arg,
        default=False,
        help="After training, merge the adapter into the base and save a 4-bit merged "
        "checkpoint under <output_dir>/merged_4bit/ for direct vLLM/LMI deployment. "
        "Requires ~96 GB GPU (e.g. ml.g7e.2xlarge). The raw adapter is still saved "
        "under <output_dir>/adapter/.",
    )
    return p.parse_args()


def parse_llama2_text(text: str) -> list[tuple[str, str]] | None:
    """Convert a Llama-2-formatted training string into [(user, assistant), ...].

    Folds optional system block into the first user turn (Gemma rejects system role).
    Returns None if no [INST] block parses.
    """
    cleaned = text.replace("<s>", "").replace("</s>", "").strip()
    turns: list[tuple[str, str]] = []
    matched = False
    for m in LLAMA2_TURN_RE.finditer(cleaned):
        matched = True
        sys_text = (m.group("sys") or "").strip()
        user_text = (m.group("user") or "").strip()
        assistant_text = (m.group("assistant") or "").strip()
        if not user_text or not assistant_text:
            return None
        if sys_text and not turns:
            user_text = f"{sys_text}\n\n{user_text}"
        turns.append((user_text, assistant_text))
    return turns if matched else None


def render_gemma4(turns: list[tuple[str, str]]) -> str:
    return "".join(GEMMA4_TURN.format(user=u, assistant=a) for u, a in turns)


def load_and_format_dataset(dataset_id: str, max_drop_pct: float) -> Dataset:
    """Download, parse Llama 2 markers, render in Gemma 4 chat format."""
    print(f"Loading dataset: {dataset_id}", flush=True)
    raw = load_dataset(dataset_id, split="train")
    if "text" not in raw.column_names:
        raise SystemExit(f"dataset has columns {raw.column_names}; expected 'text'")

    rendered: list[str] = []
    drops = 0
    for row in raw:
        turns = parse_llama2_text(row["text"])
        if turns is None:
            drops += 1
            continue
        rendered.append(render_gemma4(turns))
    total = len(raw)
    drop_pct = 100.0 * drops / max(total, 1)
    print(f"Dataset: {total} rows | parse_drops: {drops} | kept: {len(rendered)}", flush=True)
    if drop_pct > max_drop_pct:
        raise SystemExit(f"parse-drop rate {drop_pct:.2f}% exceeds budget {max_drop_pct}%")
    if not rendered:
        raise SystemExit("zero rows kept")
    return Dataset.from_dict({"text": rendered})


def find_loss_decrease(log_history: list[dict]) -> tuple[float, float] | None:
    """Return (early_loss, late_loss) for the loss-decrease assertion. None if too few entries."""
    losses = [(rec.get("step", 0), rec["loss"]) for rec in log_history if "loss" in rec]
    if len(losses) < 20:
        return None
    early = sum(loss for _, loss in losses[:5]) / 5
    late = sum(loss for _, loss in losses[-5:]) / 5
    return early, late


def main() -> int:
    args = parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        hf_login(hf_token, add_to_git_credential=False)

    dataset = load_and_format_dataset(args.dataset, args.max_drop_pct)

    print(f"Loading model: {args.model_id}", flush=True)
    # Pass model_id positionally — Unsloth has renamed the kwarg historically
    # (model_name vs model_name_or_path); positional avoids the drift.
    model, tokenizer = FastModel.from_pretrained(
        args.model_id,
        load_in_4bit=True,
        max_seq_length=args.max_seq_length,
    )

    attn_impl = getattr(model.config, "_attn_implementation", "<unknown>")
    print(f"Attention impl: {attn_impl}", flush=True)
    if attn_impl == "flash_attention_2":
        print("ERROR: Flash Attention 2 leaked through; Gemma 4 head_dim=512 will crash.", file=sys.stderr)
        return 3

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        vram_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"VRAM after load: {vram_gb:.2f} GB", flush=True)
        if vram_gb >= 45.0:
            print(f"ERROR: VRAM after load too high: {vram_gb:.2f} GB", file=sys.stderr)
            return 3

    # Token-level length filter. Gemma 4 returns a Gemma4Processor (multimodal),
    # not a tokenizer — its __call__ expects images/text/videos kwargs, and it
    # has no .encode(). Reach inside to the underlying text tokenizer.
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    before = len(dataset)
    dataset = dataset.filter(
        lambda r: len(text_tok.encode(r["text"], add_special_tokens=False)) <= args.max_seq_length,
        num_proc=1,
    )
    print(
        f"Dataset rows after token-length filter: {len(dataset)} "
        f"(dropped {before - len(dataset)} > {args.max_seq_length} tokens)",
        flush=True,
    )

    model = FastModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=GEMMA4_LORA_TARGETS,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        bf16=True,
        optim="adamw_8bit",
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=3407,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_cfg,
        processing_class=tokenizer,
    )

    print("Starting training", flush=True)
    trainer.train()

    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak VRAM during training: {peak_vram:.2f} GB", flush=True)

    pair = find_loss_decrease(trainer.state.log_history)
    if pair is not None:
        early, late = pair
        print(f"Loss early avg: {early:.4f} | late avg: {late:.4f}", flush=True)
        if late >= 0.9 * early:
            print(
                f"ERROR: loss did not decrease ≥10% (early={early:.4f}, late={late:.4f}).",
                file=sys.stderr,
            )
            return 3
    else:
        print("Skipping loss-decrease check: <20 logged loss steps.", flush=True)

    if args.merge:
        adapter_dir = os.path.join(args.output_dir, "adapter")
        merged_dir = os.path.join(args.output_dir, "merged_4bit")
        print(f"Saving adapter to {adapter_dir}", flush=True)
        trainer.model.save_pretrained(adapter_dir)
        # Unsloth's layer-by-layer merge+requantize. "merged_4bit_forced" dequantizes
        # each layer to bf16, merges the LoRA delta, then re-quantizes back to NF4 —
        # the "forced" path skips the safety check that blocks merging into a
        # 4-bit base. Output is a self-contained NF4 checkpoint vLLM can serve
        # directly with `option.quantization=bitsandbytes`. No adapter loader needed.
        print(f"Merging adapter and saving 4-bit checkpoint to {merged_dir}", flush=True)
        trainer.model.save_pretrained_merged(
            merged_dir, tokenizer, save_method="merged_4bit_forced"
        )
        print(f"Adapter files: {sorted(os.listdir(adapter_dir))}", flush=True)
        print(f"Merged files: {sorted(os.listdir(merged_dir))}", flush=True)
    else:
        print(f"Saving adapter to {args.output_dir}", flush=True)
        trainer.model.save_pretrained(args.output_dir)
        print(f"Adapter files: {sorted(os.listdir(args.output_dir))}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
