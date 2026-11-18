# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License").

"""Evaluate one locked experiment without changing its task prompts."""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.util
import json
from pathlib import Path

from reproduce import manifest, read_json, write_json


def load_dataset_snapshot(*, directory, **_kwargs):
    from datasets import load_from_disk

    return load_from_disk(directory)


def task_configs(entry, root, lock):
    from lm_eval.tasks import TaskManager

    manager = TaskManager()
    configs = []
    datasets_metadata = {}
    for name in manifest()["suites"][entry["suite"]]:
        config = manager._get_config(name)
        source = Path(manager.task_index[name]["yaml_path"])
        expected = lock["tasks"][name]
        if (
            hashlib.sha256(source.read_bytes()).hexdigest()
            != expected["task_yaml_sha256"]
        ):
            raise ValueError(f"{name}: lm-eval task YAML differs from assets lock")
        directory = root / name
        if read_json(directory / "replay-lock.json") != expected:
            raise ValueError(f"{name}: prepared dataset differs from assets lock")
        loader = functools.partial(load_dataset_snapshot, directory=str(directory))
        config["custom_dataset"] = loader
        config["dataset_kwargs"] = {}
        configs.append(config)
        dataset = loader()
        datasets_metadata[name] = {
            split: {"rows": len(data), "fingerprint": data._fingerprint}
            for split, data in dataset.items()
        }
    return configs, datasets_metadata


def hf_model(args, entry):
    from lm_eval.models.huggingface import HFLM

    model = HFLM(
        pretrained=args.model,
        dtype="bfloat16",
        device="cuda:0",
        batch_size=8,
        attn_implementation="sdpa",
    )
    if args.mode == "loop":
        source = Path(args.paper_source)
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != args.paper_sha256:
            raise ValueError("Archived author patch SHA256 does not match")
        spec = importlib.util.spec_from_file_location("paper_loop_qwen3", source)
        patch_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(patch_module)
        config = entry["loop_config"]
        patch_module.patch_qwen3_with_loop(
            model.model,
            loop_indices=list(range(config["start"], config["end"])),
            K=config["k"],
            strategy=config["strategy"],
            anchor_beta=config["anchor_beta"],
            cache_strategy=config["cache_strategy"],
            decode_mode=config["decode_mode"],
        )
    return model


def json_default(value):
    if callable(value):
        return "<locked-dataset-loader>"
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)}")


def main():
    from lm_eval import evaluator

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["experiment", "mode", "model", "data", "lock", "output", "backend"]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--port", type=int)
    parser.add_argument("--paper-source")
    parser.add_argument("--paper-sha256")
    parser.add_argument("--config-overrides")
    args = parser.parse_args()
    entry = next(
        item for item in manifest()["experiments"] if item["id"] == args.experiment
    )
    if args.config_overrides:
        overrides = read_json(args.config_overrides)
        if entry["id"] in overrides:
            entry = dict(
                entry,
                loop_config=overrides[entry["id"]],
                provenance="reconstructed-override",
            )
    settings = manifest()["evaluation"]
    configs, metadata = task_configs(entry, Path(args.data), read_json(args.lock))
    out = Path(args.output)
    write_json(out / "dataset-fingerprints.json", metadata)
    kwargs = {}
    if args.backend == "sglang":
        kwargs.update(
            model="local-completions",
            model_args={
                "model": args.model,
                "base_url": f"http://127.0.0.1:{args.port}/v1/completions",
                "tokenized_requests": False,
                "verify_certificate": False,
                "tokenizer_backend": "huggingface",
                "max_length": settings["client_max_length"],
                "num_concurrent": 1,
                "max_retries": 3,
            },
        )
    else:
        kwargs["model"] = hf_model(args, entry)
    result = evaluator.simple_evaluate(
        **kwargs,
        tasks=configs,
        num_fewshot=settings["num_fewshot"],
        batch_size=settings["batch_size"],
        limit=entry["limit"],
        apply_chat_template=False,
        log_samples=True,
        random_seed=0,
        numpy_random_seed=1234,
        torch_random_seed=1234,
        fewshot_random_seed=1234,
    )
    if result is None:
        raise RuntimeError("lm-eval returned no result")
    samples = result.pop("samples")
    for name, records in samples.items():
        with (out / f"samples-{name}.jsonl").open("w") as stream:
            for record in records:
                stream.write(json.dumps(record, default=json_default) + "\n")
    (out / "results.json").write_text(
        json.dumps(result, indent=2, default=json_default) + "\n"
    )
    expected = set(manifest()["suites"][entry["suite"]])
    actual = set(result["results"])
    if actual != expected:
        raise RuntimeError(f"Task mismatch: expected {expected}, got {actual}")
    for name in expected:
        record = result["n-samples"][name]
        expected_count = (
            min(entry["limit"], record["original"])
            if entry["limit"] is not None
            else record["original"]
        )
        if record["effective"] != expected_count:
            raise RuntimeError(f"{name}: unexpected sample count {record}")
    print(json.dumps(result["results"], indent=2, default=json_default))


if __name__ == "__main__":
    main()
