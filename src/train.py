"""LoRA fine-tuning of Qwen3-8B with DDP. Launched by torchrun inside a Kubeflow TrainJob."""
import contextlib
import json
import math
import os
import sys
import time

import boto3
import mlflow
import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import build_messages, target_text  # noqa: E402

CFG = {
    "base_model": os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B"),
    "epochs": int(os.environ.get("EPOCHS", "1")),
    "lr": float(os.environ.get("LR", "2e-4")),
    "micro_batch": int(os.environ.get("MICRO_BATCH", "4")),
    "grad_accum": int(os.environ.get("GRAD_ACCUM", "4")),
    "max_len": int(os.environ.get("MAX_LEN", "1024")),
    "lora_r": int(os.environ.get("LORA_R", "16")),
    "lora_alpha": int(os.environ.get("LORA_ALPHA", "32")),
    "warmup_steps": int(os.environ.get("WARMUP_STEPS", "10")),
}
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

def s3_split(uri):
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    return bucket, key

def read_jsonl(uri):
    bucket, key = s3_split(uri)
    body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read().decode()
    return [json.loads(line) for line in body.splitlines() if line.strip()]

def upload_dir(local_dir, uri):
    bucket, prefix = s3_split(uri)
    s3 = boto3.client("s3")
    for name in os.listdir(local_dir):
        s3.upload_file(os.path.join(local_dir, name), bucket, f"{prefix.rstrip('/')}/{name}")

def encode(tok, row):
    # Same chat template as inference: thinking disabled, answer followed by <|im_end|>.
    prompt = tok.apply_chat_template(build_messages(row["question"], row["contexts"]),
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tok(target_text(row) + "<|im_end|>", add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(answer_ids) > CFG["max_len"]:
        return None
    return {"input_ids": prompt_ids + answer_ids, "labels": [-100] * len(prompt_ids) + answer_ids}

def collate(batch, pad_id):
    width = max(len(x["input_ids"]) for x in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, x in enumerate(batch):
        n = len(x["input_ids"])
        ids[i, :n] = torch.tensor(x["input_ids"])
        labels[i, :n] = torch.tensor(x["labels"])
        mask[i, :n] = 1
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}

def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.manual_seed(42)
    prefix = os.environ["S3_PREFIX"]

    tok = AutoTokenizer.from_pretrained(CFG["base_model"])
    rows = read_jsonl(f"{prefix}/data/train.jsonl")
    data = [e for e in (encode(tok, r) for r in rows) if e is not None]

    model = AutoModelForCausalLM.from_pretrained(CFG["base_model"], dtype=torch.bfloat16,
                                                 device_map={"": local_rank})
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"],
                                             lora_dropout=0.05, target_modules=LORA_TARGETS,
                                             task_type="CAUSAL_LM"))
    ddp = DDP(model, device_ids=[local_rank])

    sampler = DistributedSampler(data, num_replicas=world, rank=rank, shuffle=True, seed=42)
    loader = DataLoader(data, batch_size=CFG["micro_batch"], sampler=sampler,
                        collate_fn=lambda b: collate(b, tok.pad_token_id))

    trainable = [p for p in ddp.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=CFG["lr"], weight_decay=0.0)
    total_steps = CFG["epochs"] * (len(loader) // CFG["grad_accum"])

    def lr_lambda(step):  # linear warmup, then cosine decay
        if step < CFG["warmup_steps"]:
            return (step + 1) / CFG["warmup_steps"]
        progress = (step - CFG["warmup_steps"]) / max(1, total_steps - CFG["warmup_steps"])
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    if rank == 0:
        mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"], log_system_metrics=True)
        mlflow.log_params({**CFG, "world_size": world, "train_examples": len(data),
                           "skipped_too_long": len(rows) - len(data), "total_steps": total_steps,
                           "trainable_params": sum(p.numel() for p in trainable)})

    step, start = 0, time.time()
    ddp.train()
    for epoch in range(CFG["epochs"]):
        sampler.set_epoch(epoch)
        opt.zero_grad(set_to_none=True)
        run_loss = torch.zeros(1, device="cuda")
        run_tokens = torch.zeros(1, device="cuda")
        t0 = time.time()
        for i, batch in enumerate(loader):
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            sync = (i + 1) % CFG["grad_accum"] == 0
            # Skip the gradient all-reduce on accumulation micro-steps.
            with contextlib.nullcontext() if sync else ddp.no_sync():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = ddp(**batch).loss / CFG["grad_accum"]
                loss.backward()
            run_loss += loss.detach()
            run_tokens += batch["attention_mask"].sum()
            if not sync:
                continue

            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            stats = torch.cat([run_loss, run_tokens])
            dist.all_reduce(stats)
            if rank == 0:
                metrics = {"train_loss": stats[0].item() / world, "lr": sched.get_last_lr()[0],
                           "tokens_per_sec": stats[1].item() / (time.time() - t0)}
                mlflow.log_metrics(metrics, step=step)
                print(f"step {step}/{total_steps} " + " ".join(f"{k}={v:.4g}" for k, v in metrics.items()),
                      flush=True)
            run_loss.zero_()
            run_tokens.zero_()
            t0 = time.time()

    if rank == 0:
        out_dir = "/tmp/adapter"
        ddp.module.save_pretrained(out_dir)               # adapter_config.json + adapter_model.safetensors
        upload_dir(out_dir, f"{prefix}/adapter")          # for evaluation and serving
        mlflow.log_artifacts(out_dir, artifact_path="adapter")
        mlflow.log_metric("train_runtime_sec", time.time() - start)
        mlflow.end_run()
    dist.barrier(device_ids=[local_rank])
    dist.destroy_process_group()

if __name__ == "__main__":
    try:
        main()
    except Exception:
        if mlflow.active_run():
            mlflow.end_run(status="FAILED")
        raise
