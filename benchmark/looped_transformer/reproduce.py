# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License").

"""Portable replay of the looped-transformer experiment matrix."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE_FILES = (
    "python/sglang/srt/arg_groups/fields/model.py",
    "python/sglang/srt/models/looped_transformer.py",
    "python/sglang/srt/models/qwen2.py",
    "python/sglang/srt/models/qwen3.py",
    "python/sglang/srt/models/llama.py",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest():
    return read_json(HERE / "experiments.json")


def selected(names):
    experiments = manifest()["experiments"]
    if not names or names == ["all"]:
        return experiments
    unknown = set(names) - {entry["id"] for entry in experiments}
    if unknown:
        raise ValueError(f"Unknown experiment IDs: {sorted(unknown)}")
    return [entry for entry in experiments if entry["id"] in names]


def capture(command):
    return subprocess.check_output(command, cwd=REPO, text=True).strip()


def inventory():
    return {
        "python": sys.version,
        "platform": sys.platform,
        "git_head": capture(["git", "rev-parse", "HEAD"]),
        "git_status": capture(["git", "status", "--porcelain"]),
        "source_sha256": {name: digest(REPO / name) for name in SOURCE_FILES},
        "pip_freeze": capture([sys.executable, "-m", "pip", "freeze", "--all"]),
        "nvidia_smi": capture(["nvidia-smi"]),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def lock_assets(args):
    from huggingface_hub import HfApi
    from lm_eval.tasks import TaskManager

    entries = selected(args.experiments)
    task_manager = TaskManager()
    api = HfApi()
    lock = {"schema_version": 1, "models": {}, "tasks": {}}
    lock["resolved_on"] = time.strftime("%Y-%m-%d", time.gmtime())
    lock["historical_revisions_recovered"] = False
    for model in sorted({entry["model"] for entry in entries}):
        lock["models"][model] = api.model_info(model).sha
    task_names = {
        task for entry in entries for task in manifest()["suites"][entry["suite"]]
    }
    for name in sorted(task_names):
        config = task_manager._get_config(name)
        if "class" in config or config.get("custom_dataset"):
            raise ValueError(f"{name}: custom/Python dataset needs a separate lock")
        kwargs = dict(config.get("dataset_kwargs") or {})
        repo_id = config["dataset_path"]
        revision = api.dataset_info(repo_id, revision=kwargs.get("revision")).sha
        lock["tasks"][name] = {
            "dataset": repo_id,
            "name": config.get("dataset_name"),
            "revision": revision,
            "kwargs": kwargs,
            "task_yaml_sha256": digest(task_manager.task_index[name]["yaml_path"]),
        }
    Path(args.lock).parent.mkdir(parents=True, exist_ok=True)
    if Path(args.lock).exists():
        raise FileExistsError(
            "Lock already exists; choose a new filename to re-resolve"
        )
    write_json(args.lock, lock)
    print(f"Locked {len(lock['models'])} models and {len(lock['tasks'])} tasks")


def prepare_data(args):
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    lock = read_json(args.lock)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if not args.datasets_only:
        for repo_id, revision in lock["models"].items():
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                allow_patterns=[
                    "*.json",
                    "*.jinja",
                    "*.model",
                    "*.txt",
                    "*.safetensors",
                ],
            )
    for name, spec in lock["tasks"].items():
        dest = root / "datasets" / name
        if dest.exists():
            if (dest / "replay-lock.json").exists() and read_json(
                dest / "replay-lock.json"
            ) == spec:
                continue
            raise FileExistsError(
                f"{dest}: incomplete/mismatched preparation directory"
            )
        kwargs = dict(spec["kwargs"])
        kwargs.pop("revision", None)
        dataset = load_dataset(
            path=spec["dataset"],
            name=spec["name"],
            revision=spec["revision"],
            **kwargs,
        )
        dataset.save_to_disk(str(dest))
        write_json(dest / "replay-lock.json", spec)
    write_json(root / "assets.lock.json", lock)


def server_command(entry, mode, model, port):
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tp-size",
        "1",
        "--dtype",
        "bfloat16",
        "--log-level",
        "warning",
    ]
    if mode == "loop":
        if entry["loop_config"] is None:
            raise ValueError(
                f"{entry['id']}: historical loop configuration was not archived; "
                "supply --config-overrides and label the result reconstructed"
            )
        command += [
            "--loop-config",
            json.dumps(entry["loop_config"], separators=(",", ":")),
        ]
    return command


def wait_ready(process, port, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Server exited with code {process.returncode}; see server.log"
            )
        try:
            with urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError):
            pass
        time.sleep(2)
    raise TimeoutError(f"Health check timed out after {timeout}s")


def stop_owned_process(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    if process.poll() is None:
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def check_port(port):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))


def check_source():
    commit = manifest()["implementation_commit"]
    for name in SOURCE_FILES:
        original = subprocess.check_output(
            ["git", "show", f"{commit}:{name}"], cwd=REPO
        )
        if hashlib.sha256(original).hexdigest() != digest(REPO / name):
            raise RuntimeError(
                f"{name} differs from the recorded implementation commit"
            )


def check_environment():
    from importlib.metadata import version

    import torch

    import sglang

    if version("lm-eval") != "0.4.11":
        raise RuntimeError("Replay requires lm-eval==0.4.11")
    expected = (REPO / "python" / "sglang").resolve()
    if Path(sglang.__file__).resolve().parent != expected:
        raise RuntimeError("Importing another SGLang checkout; set PYTHONPATH=python")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA runtime and an available NVIDIA GPU are required")


def run_one(args, entry, mode):
    from huggingface_hub import snapshot_download

    root = Path(args.output)
    dest = root / entry["id"] / args.backend / mode
    dest.mkdir(parents=True, exist_ok=False)
    status = {"status": "running", "experiment": entry["id"], "mode": mode}
    write_json(dest / "status.json", status)
    try:
        lock = read_json(args.lock)
        model = snapshot_download(
            repo_id=entry["model"],
            revision=lock["models"][entry["model"]],
            local_files_only=True,
        )
        write_json(dest / "environment.json", inventory())
        write_json(dest / "experiment.json", entry)
        write_json(dest / "assets.lock.json", lock)
        command = [
            sys.executable,
            str(HERE / "evaluate.py"),
            "--experiment",
            entry["id"],
            "--mode",
            mode,
            "--model",
            model,
            "--data",
            str(root / "datasets"),
            "--lock",
            args.lock,
            "--output",
            str(dest),
            "--port",
            str(args.port),
            "--backend",
            args.backend,
        ]
        if args.config_overrides:
            command += ["--config-overrides", args.config_overrides]
        if args.backend == "paper-hf" and mode == "loop":
            command += [
                "--paper-source",
                args.paper_source,
                "--paper-sha256",
                args.paper_sha256,
            ]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
        environment["PYTHONPATH"] = str(REPO / "python")
        started = time.monotonic()
        with contextlib.ExitStack() as stack:
            if args.backend == "sglang":
                check_port(args.port)
                serve = server_command(entry, mode, model, args.port)
                write_json(
                    dest / "commands.json", {"server": serve, "evaluation": command}
                )
                server_log = stack.enter_context((dest / "server.log").open("w"))
                process = subprocess.Popen(
                    serve,
                    cwd=REPO,
                    env=environment,
                    stdout=server_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                stack.callback(stop_owned_process, process)
                wait_ready(process, args.port, args.startup_timeout)
            else:
                write_json(dest / "commands.json", {"evaluation": command})
            with (dest / "eval.log").open("w") as log:
                evaluation = subprocess.Popen(
                    command,
                    cwd=REPO,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                stack.callback(stop_owned_process, evaluation)
                if evaluation.wait() != 0:
                    raise subprocess.CalledProcessError(evaluation.returncode, command)
        status.update(status="complete", elapsed_seconds=time.monotonic() - started)
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(dest / "status.json", status)
    print(f"Complete: {entry['id']} / {args.backend} / {mode}", flush=True)


def run(args):
    entries = selected(args.experiments)
    if args.config_overrides:
        overrides = read_json(args.config_overrides)
        entries = [
            dict(
                entry,
                loop_config=overrides[entry["id"]],
                provenance="reconstructed-override",
            )
            if entry["id"] in overrides
            else entry
            for entry in entries
        ]
    if args.dry_run:
        for entry in entries:
            for mode in args.modes:
                print(
                    json.dumps(
                        {
                            "id": entry["id"],
                            "mode": mode,
                            "backend": args.backend,
                            "tasks": manifest()["suites"][entry["suite"]],
                            "limit": entry["limit"],
                            "loop_config": entry["loop_config"],
                            "blocked": mode == "loop" and entry["loop_config"] is None,
                        }
                    )
                )
        return
    for entry in entries:
        if "loop" in args.modes:
            server_command(entry, "loop", "<locked-model>", args.port)
        if args.backend == "paper-hf" and not entry["model"].startswith("Qwen/Qwen3"):
            raise ValueError("The archived author patch only implements Qwen3")
    if args.backend == "paper-hf" and "loop" in args.modes:
        if not args.paper_source or not args.paper_sha256:
            raise ValueError("paper-hf loop requires --paper-source and --paper-sha256")
    check_source()
    check_environment()
    for entry in entries:
        for mode in args.modes:
            run_one(args, entry, mode)


def summarize(args):
    root = Path(args.output)
    rows = []
    for path in sorted(root.glob("*/*/baseline/results.json")):
        loop_path = path.parent.parent / "loop" / "results.json"
        if not loop_path.exists():
            continue
        status_paths = [path.parent / "status.json", loop_path.parent / "status.json"]
        if any(
            not item.exists() or read_json(item)["status"] != "complete"
            for item in status_paths
        ):
            continue
        base = read_json(path)["results"]
        loop = read_json(loop_path)["results"]
        for task, metrics in base.items():
            for metric, before in metrics.items():
                if (
                    "_stderr" in metric
                    or metric == "alias"
                    or not isinstance(before, (int, float))
                ):
                    continue
                after = loop[task][metric]
                error_key = metric.replace(",", "_stderr,", 1)
                rows.append(
                    {
                        "experiment": path.parts[-4],
                        "backend": path.parts[-3],
                        "task": task,
                        "metric": metric,
                        "baseline": before,
                        "loop": after,
                        "delta": after - before,
                        "baseline_stderr": metrics.get(error_key),
                        "loop_stderr": loop[task].get(error_key),
                    }
                )
    write_json(root / "summary.json", rows)
    historical = read_json(HERE / "historical-results.json")["records"]
    reference = {
        (item["experiment"], item["backend"], task_metric): values
        for item in historical
        for task_metric, values in item["metrics"].items()
    }
    comparisons = []
    for row in rows:
        task_metric = row["task"] + "/" + row["metric"].split(",")[0]
        key = (row["experiment"], row["backend"], task_metric)
        if key in reference:
            target = reference[key]
            comparisons.append(
                dict(
                    experiment=key[0],
                    backend=key[1],
                    metric=task_metric,
                    baseline_difference=row["baseline"] - target["baseline"],
                    loop_difference=row["loop"] - target["loop"],
                )
            )
    write_json(root / "historical-differences.json", comparisons)
    lines = [
        "| Experiment | Backend | Task | Metric | Baseline | Loop | Delta |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['backend']} | {row['task']} | {row['metric']} "
            f"| {row['baseline']:.8f} | {row['loop']:.8f} | {row['delta']:+.8f} |"
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(rows)} metric comparisons")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["list", "lock", "prepare", "run", "summarize"]
    )
    parser.add_argument("--experiments", nargs="+")
    parser.add_argument("--output", default="artifacts/looped-transformer")
    parser.add_argument(
        "--lock", default="benchmark/looped_transformer/assets.lock.json"
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--port", type=int, default=30100)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument(
        "--modes", nargs="+", choices=["baseline", "loop"], default=["baseline", "loop"]
    )
    parser.add_argument("--backend", choices=["sglang", "paper-hf"], default="sglang")
    parser.add_argument("--paper-source")
    parser.add_argument("--paper-sha256")
    parser.add_argument("--config-overrides")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--datasets-only", action="store_true")
    args = parser.parse_args()
    if args.action == "list":
        for entry in selected(args.experiments):
            print(json.dumps(entry))
    else:

        def interrupted(_signum, _frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupted)
        {
            "lock": lock_assets,
            "prepare": prepare_data,
            "run": run,
            "summarize": summarize,
        }[args.action](args)


if __name__ == "__main__":
    main()
