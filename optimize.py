import os
import subprocess
import time
import json
import requests
import optuna
from hydra import main
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

@main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Load configuration values
    model_path = cfg.server.model_path
    docker_image = cfg.container.image
    port = cfg.container.port

    health_url = cfg.healthcheck.url
    health_retries = cfg.healthcheck.max_tries
    health_interval = cfg.healthcheck.interval

    root_dir = get_original_cwd()
    benchmark_script = os.path.join(root_dir, "benchmark_serving.py")
    benchmark_cmd = [
        "python3", benchmark_script,
        "--random-input-len", str(cfg.benchmark.input_len),
        "--random-output-len", str(cfg.benchmark.output_len),
        "--model", cfg.server.model,
        "--base-url", cfg.server.base_url,
        "--tokenizer", model_path,
        "--dataset-name", cfg.benchmark.dataset_name,
        "--percentile-metrics", cfg.benchmark.percentile_metrics,
        "--save-result",
        "--result-filename", cfg.benchmark.result_filename,
        "--max-concurrency", str(cfg.benchmark.max_concurrency),
        "--num-prompts", str(cfg.benchmark.num_prompts)
    ]

    def run_container(chunk_size: int, max_tokens: int, policy: str, conserv: float) -> str:
        name = cfg.container.name
        # Remove any existing container
        subprocess.run(["docker", "rm", "-f", name], stderr=subprocess.DEVNULL)

        cmd = [
            "docker", "run", "--gpus", "all",
            "-p", f"{port}:{port}",
            "-v", f"{model_path}:{model_path}",
            "--ipc=host",
            "--rm", "--name", name,
            docker_image,
            "python3", "-m", "sglang.launch_server",
            "--model-path", model_path,
            "--host", "0.0.0.0",
            "--port", str(port),
            "--chunked-prefill-size", str(chunk_size),
            "--max-prefill-tokens", str(max_tokens),
            "--schedule-policy", policy,
            "--schedule-conservativeness", str(conserv),
            "--served-model-name", cfg.server.model
        ]
        print("Starting container:", " ".join(cmd))
        subprocess.Popen(cmd)
        return name

    def stop_container(name: str) -> None:
        subprocess.run(["docker", "stop", name], stderr=subprocess.DEVNULL)
        time.sleep(5)
        subprocess.run(["docker", "rm", "-f", name], stderr=subprocess.DEVNULL)

    def wait_for_healthcheck(url: str) -> None:
        for attempt in range(health_retries):
            try:
                if requests.get(url, timeout=60).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(health_interval)
        raise RuntimeError("Health check timed out")

    def measure_latency() -> float:
        try:
            print(f"[DEBUG] Running benchmark: {' '.join(benchmark_cmd)}")
            proc = subprocess.run(
                benchmark_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300
            )
            print("[DEBUG] stdout:", proc.stdout)
            print("[DEBUG] stderr:", proc.stderr)

            result_path = os.path.join(os.getcwd(), cfg.benchmark.result_filename)
            print(f"[DEBUG] Checking result file: {result_path}, exists: {os.path.exists(result_path)}")

            with open(result_path, 'r') as f:
                data = json.load(f)

            return float(data["mean_e2el_ms"])

        except subprocess.TimeoutExpired:
            raise RuntimeError("Benchmark script timed out")
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Error parsing benchmark result: {e}")

    def objective(trial: optuna.Trial) -> float:
        if trial.number == 0:
            # Use default parameters for the first trial
            chunk_size = cfg.optuna.default.chunked_prefill_size
            max_tokens = cfg.optuna.default.max_prefill_tokens
            policy = cfg.optuna.default.schedule_policy
            conserv = cfg.optuna.default.schedule_conservativeness
        else:
            # Suggest parameters for subsequent trials
            chunk_size = trial.suggest_int("chunked_prefill_size", *cfg.optuna.chunked_prefill_size)
            max_tokens = trial.suggest_int("max_prefill_tokens", *cfg.optuna.max_prefill_tokens)
            policy = trial.suggest_categorical("schedule_policy", cfg.optuna.schedule_policy)
            low, high, step = cfg.optuna.schedule_conservativeness
            conserv = trial.suggest_float("schedule_conservativeness", low, high, step=step)

        container_name = None
        try:
            container_name = run_container(chunk_size, max_tokens, policy, conserv)
            wait_for_healthcheck(health_url)
            return measure_latency()
        except Exception as e:
            print(f"[WARNING] Trial {trial.number} failed: {e}")
            return float("inf")
        finally:
            if container_name:
                stop_container(container_name)

    # Set up Optuna study
    import logging
    logging.basicConfig(level=logging.INFO)
    study = optuna.create_study(
        direction="minimize",
        study_name="sglang_optuna",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(),
        storage=cfg.optuna.storage
    )
    study.optimize(objective, n_trials=cfg.optuna.n_trials)

    print("Best trial parameters:", study.best_trial.params)
    print("Best trial latency:", study.best_trial.value)

if __name__ == "__main__":
    main()