"""Qwen3-8B RAFT-style LoRA pipeline: prepare -> train (TrainJob) -> evaluate -> promote."""
import os
from typing import NamedTuple

from kfp import compiler, dsl, kubernetes
from kfp.dsl import Metrics, Output

VLLM_IMAGE = "vllm/vllm-openai:v0.33.0"
PY_IMAGE = "python:3.12-slim"
MLFLOW_PKG = "mlflow-skinny==3.18.0"

@dsl.component(base_image=PY_IMAGE, packages_to_install=["datasets", "boto3", MLFLOW_PKG])
def prepare_data(bucket: str, region: str, mlflow_uri: str, experiment: str, base_model: str,
                 n_train: int, n_eval: int, num_distractors: int, kfp_run_id: str) -> str:
    """Convert SQuAD v2 to RAFT format (gold paragraph + distractors) and open an MLflow run."""
    import json
    import random

    import boto3
    import mlflow
    from datasets import load_dataset

    rnd = random.Random(42)
    ds = load_dataset("rajpurkar/squad_v2")

    def unique_contexts(split):
        seen = {}
        for title, ctx in zip(ds[split]["title"], ds[split]["context"]):
            seen.setdefault(ctx, title)
        return [(title, ctx) for ctx, title in seen.items()]

    def build(split, n, pool):
        rows = []
        for r in ds[split].shuffle(seed=42).select(range(n)):
            distractors = []
            while len(distractors) < num_distractors:   # paragraphs from other articles only
                title, ctx = rnd.choice(pool)
                if title != r["title"] and ctx not in distractors:
                    distractors.append(ctx)
            contexts = [r["context"], *distractors]
            rnd.shuffle(contexts)
            rows.append({"id": r["id"], "question": r["question"], "contexts": contexts,
                         "gold_context": r["context"],
                         "answers": list(dict.fromkeys(r["answers"]["text"]))})  # [] = unanswerable
        return rows

    train_rows = build("train", n_train, unique_contexts("train"))
    val_pool = unique_contexts("validation")
    eval_rows = build("validation", n_eval, val_pool)
    corpus = [{"doc_id": f"val-{i:04d}", "title": t, "text": c} for i, (t, c) in enumerate(val_pool)]
    doc_id_of = {d["text"]: d["doc_id"] for d in corpus}
    for r in eval_rows:
        r["gold_doc_id"] = doc_id_of[r["gold_context"]]

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"kfp-{kfp_run_id[:8]}") as run:
        mlflow.set_tags({"kfp_run_id": kfp_run_id, "base_model": base_model})
        mlflow.log_params({
            "dataset": "rajpurkar/squad_v2", "n_train": len(train_rows), "n_eval": len(eval_rows),
            "num_distractors": num_distractors, "corpus_docs": len(corpus),
            "train_unanswerable_ratio": round(sum(not r["answers"] for r in train_rows) / len(train_rows), 3),
        })
        run_id = run.info.run_id

    s3 = boto3.client("s3", region_name=region)

    def put_jsonl(key, rows):
        s3.put_object(Bucket=bucket, Key=key,
                      Body="\n".join(json.dumps(r, ensure_ascii=False) for r in rows).encode())

    put_jsonl(f"runs/{run_id}/data/train.jsonl", train_rows)
    put_jsonl(f"runs/{run_id}/data/eval.jsonl", eval_rows)
    put_jsonl("datasets/squad_v2/corpus.jsonl", corpus)          # RAG corpus (Lab 6)
    print(f"run_id={run_id} train={len(train_rows)} eval={len(eval_rows)} corpus={len(corpus)}")
    return run_id

@dsl.component(base_image=PY_IMAGE, packages_to_install=["kubernetes"])
def train_lora(run_id: str, bucket: str, region: str, mlflow_uri: str, image: str,
               base_model: str, namespace: str, num_nodes: int, epochs: int, lr: float,
               micro_batch: int, grad_accum: int) -> str:
    """Create a Kubeflow Trainer TrainJob and block until it completes."""
    import time

    from kubernetes import client, config
    from kubernetes.client.exceptions import ApiException

    config.load_incluster_config()
    api = client.CustomObjectsApi()
    group, version, plural = "trainer.kubeflow.org", "v1alpha1", "trainjobs"
    name = f"qwen3-lora-{run_id[:12]}"
    prefix = f"s3://{bucket}/runs/{run_id}"

    env = {
        "BASE_MODEL": base_model, "EPOCHS": str(epochs), "LR": str(lr),
        "MICRO_BATCH": str(micro_batch), "GRAD_ACCUM": str(grad_accum),
        "S3_PREFIX": prefix, "AWS_DEFAULT_REGION": region,
        "MLFLOW_TRACKING_URI": mlflow_uri, "MLFLOW_RUN_ID": run_id,
        "PYTHONUNBUFFERED": "1", "PIP_BREAK_SYSTEM_PACKAGES": "1",
    }
    script = "\n".join([
        "set -euo pipefail",
        "python3 -m pip install --no-cache-dir -q 'peft==0.21.3' 'mlflow-skinny==3.18.0' "
        "boto3 psutil nvidia-ml-py",
        "python3 -m torch.distributed.run /workspace/code/train.py",
    ])
    pod_patch = {
        "serviceAccountName": "llm-workload",
        "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
        "volumes": [
            {"name": "code", "configMap": {"name": "workshop-code"}},
            {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "8Gi"}},
        ],
        "containers": [{"name": "node", "volumeMounts": [
            {"name": "code", "mountPath": "/workspace/code"},
            {"name": "dshm", "mountPath": "/dev/shm"},
        ]}],
    }
    body = {
        "apiVersion": f"{group}/{version}",
        "kind": "TrainJob",
        "metadata": {"name": name, "namespace": namespace, "labels": {"mlflow-run-id": run_id}},
        "spec": {
            "runtimeRef": {"name": "torch-distributed"},
            "trainer": {
                "image": image,
                "command": ["bash", "-c", script],
                "numNodes": num_nodes,
                "numProcPerNode": 1,
                "resourcesPerNode": {
                    "requests": {"cpu": "6", "memory": "48Gi"},
                    "limits": {"nvidia.com/gpu": "1", "memory": "56Gi"},
                },
                "env": [{"name": k, "value": v} for k, v in env.items()],
            },
            "runtimePatches": [{
                "manager": "workshop.example.com/kfp",
                "trainingRuntimeSpec": {"template": {"spec": {"replicatedJobs": [{
                    "name": "node",
                    "template": {"spec": {"template": {"spec": pod_patch}}},
                }]}}},
            }],
        },
    }

    try:
        api.create_namespaced_custom_object(group, version, namespace, plural, body)
        print(f"created TrainJob {namespace}/{name}")
    except ApiException as e:
        if e.status != 409:      # 409: already created by a previous attempt of this step
            raise
        print(f"TrainJob {name} already exists, watching it")

    while True:
        job = api.get_namespaced_custom_object(group, version, namespace, plural, name)
        conds = {c["type"]: c for c in job.get("status", {}).get("conditions", [])}
        if conds.get("Complete", {}).get("status") == "True":
            return f"{prefix}/adapter"
        if conds.get("Failed", {}).get("status") == "True":
            raise RuntimeError(f"TrainJob failed: {conds['Failed'].get('message')}")
        print("waiting:", {k: v.get("status") for k, v in conds.items()}, flush=True)
        time.sleep(60)

@dsl.component(base_image=VLLM_IMAGE, packages_to_install=[MLFLOW_PKG, "boto3"])
def evaluate(run_id: str, bucket: str, region: str, mlflow_uri: str, base_model: str,
             adapter_uri: str, min_gain: float, metrics: Output[Metrics]) -> NamedTuple(
                 "EvalOutputs", [("decision", str), ("base_overall_f1", float), ("ft_overall_f1", float)]):
    """Score base vs LoRA on the held-out set with vLLM offline inference."""
    import json
    import os
    import sys
    from collections import namedtuple

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    sys.path.insert(0, "/workspace/code")
    import boto3
    import mlflow
    from common import build_messages, score_predictions
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    s3 = boto3.client("s3", region_name=region)
    body = s3.get_object(Bucket=bucket, Key=f"runs/{run_id}/data/eval.jsonl")["Body"].read().decode()
    rows = [json.loads(line) for line in body.splitlines() if line.strip()]

    adapter_dir = "/tmp/adapter"
    os.makedirs(adapter_dir, exist_ok=True)
    a_bucket, _, a_prefix = adapter_uri.removeprefix("s3://").partition("/")
    for obj in s3.list_objects_v2(Bucket=a_bucket, Prefix=f"{a_prefix}/")["Contents"]:
        s3.download_file(a_bucket, obj["Key"], os.path.join(adapter_dir, os.path.basename(obj["Key"])))

    tok = AutoTokenizer.from_pretrained(base_model)
    prompts = [tok.apply_chat_template(build_messages(r["question"], r["contexts"]), tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False) for r in rows]
    llm = LLM(model=base_model, enable_lora=True, max_lora_rank=16, max_model_len=4096,
              gpu_memory_utilization=0.85)
    params = SamplingParams(temperature=0.0, max_tokens=64)
    base_preds = [o.outputs[0].text.strip() for o in llm.generate(prompts, params)]
    ft_preds = [o.outputs[0].text.strip() for o in llm.generate(
        prompts, params, lora_request=LoRARequest("ft", 1, adapter_dir))]

    base_m, ft_m = score_predictions(rows, base_preds), score_predictions(rows, ft_preds)
    mlflow.set_tracking_uri(mlflow_uri)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({f"eval_base_{k}": v for k, v in base_m.items()})
        mlflow.log_metrics({f"eval_ft_{k}": v for k, v in ft_m.items()})
        mlflow.log_dict({"samples": [
            {"question": r["question"], "gold": r["answers"] or ["<no answer>"], "base": b, "ft": f}
            for r, b, f in zip(rows[:50], base_preds, ft_preds)]}, "eval/samples.json")

    for k, v in base_m.items():
        metrics.log_metric(f"base_{k}", round(v, 4))
    for k, v in ft_m.items():
        metrics.log_metric(f"ft_{k}", round(v, 4))

    decision = "promote" if ft_m["overall_f1"] >= base_m["overall_f1"] + min_gain else "skip"
    print(json.dumps({"base": base_m, "ft": ft_m, "decision": decision}, indent=2))
    outputs = namedtuple("EvalOutputs", ["decision", "base_overall_f1", "ft_overall_f1"])
    return outputs(decision, base_m["overall_f1"], ft_m["overall_f1"])

@dsl.component(base_image=PY_IMAGE, packages_to_install=["boto3", "kubernetes", MLFLOW_PKG])
def promote(run_id: str, bucket: str, region: str, adapter_uri: str, mlflow_uri: str, namespace: str):
    """Point serving/CURRENT at the new adapter and roll the vLLM deployment."""
    import boto3
    import mlflow
    from kubernetes import client, config
    from kubernetes.client.exceptions import ApiException

    boto3.client("s3", region_name=region).put_object(
        Bucket=bucket, Key="serving/CURRENT", Body=adapter_uri.encode())
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.MlflowClient().set_tag(run_id, "promoted", "true")

    config.load_incluster_config()
    patch = {"spec": {"template": {"metadata": {"annotations": {"workshop/adapter-run": run_id}}}}}
    try:
        client.AppsV1Api().patch_namespaced_deployment("vllm", namespace, patch)
        print("vLLM rollout triggered")
    except ApiException as e:
        if e.status != 404:
            raise
        print("vLLM deployment not found yet; it will read serving/CURRENT on first start")

@dsl.pipeline(name="qwen3-8b-raft-lora")
def qwen3_raft_pipeline(
    bucket: str,
    region: str,
    mlflow_uri: str = "http://mlflow.mlflow.svc.cluster.local:5000",
    experiment: str = "qwen3-8b-squad-raft",
    base_model: str = "Qwen/Qwen3-8B",
    namespace: str = "llm",
    n_train: int = 4000,
    n_eval: int = 1000,
    num_distractors: int = 2,
    num_nodes: int = 1,
    epochs: int = 1,
    lr: float = 2e-4,
    micro_batch: int = 4,
    grad_accum: int = 4,
    min_gain: float = 0.02,
):
    prep = prepare_data(bucket=bucket, region=region, mlflow_uri=mlflow_uri, experiment=experiment,
                        base_model=base_model, n_train=n_train, n_eval=n_eval,
                        num_distractors=num_distractors, kfp_run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER)

    train = train_lora(run_id=prep.output, bucket=bucket, region=region, mlflow_uri=mlflow_uri,
                       image=VLLM_IMAGE, base_model=base_model, namespace=namespace,
                       num_nodes=num_nodes, epochs=epochs, lr=lr,
                       micro_batch=micro_batch, grad_accum=grad_accum)

    ev = evaluate(run_id=prep.output, bucket=bucket, region=region, mlflow_uri=mlflow_uri,
                  base_model=base_model, adapter_uri=train.output, min_gain=min_gain)
    ev.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit("1")
    ev.set_cpu_request("4").set_memory_request("32Gi")
    ev.set_env_variable("PIP_BREAK_SYSTEM_PACKAGES", "1")
    kubernetes.add_toleration(ev, key="nvidia.com/gpu", operator="Exists", effect="NoSchedule")
    kubernetes.use_config_map_as_volume(ev, config_map_name="workshop-code", mount_path="/workspace/code")

    with dsl.If(ev.outputs["decision"] == "promote"):
        pr = promote(run_id=prep.output, bucket=bucket, region=region,
                     adapter_uri=train.output, mlflow_uri=mlflow_uri, namespace=namespace)
        pr.set_caching_options(False)

    for task in (prep, train, ev):
        task.set_caching_options(False)

if __name__ == "__main__":
    import kfp

    compiler.Compiler().compile(qwen3_raft_pipeline, "qwen3_raft_pipeline.yaml")
    client = kfp.Client(host=os.environ.get("KFP_HOST", "http://localhost:8080"))
    run = client.create_run_from_pipeline_func(
        qwen3_raft_pipeline,
        arguments={"bucket": os.environ["BUCKET"], "region": os.environ["AWS_REGION"],
                   "num_nodes": int(os.environ.get("NUM_NODES", "1"))},
        experiment_name="qwen3-workshop",
        run_name="qwen3-raft-lora",
    )
    print("KFP run id:", run.run_id)
