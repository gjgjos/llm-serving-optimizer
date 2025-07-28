# ⚡️ LLM Serving Optimizer

Optimize LLM serving speed using **SGLang + Optuna + Hydra**.

- 🔧 Automatically tune `chunked-prefill-size`, `max-prefill-tokens`, `schedule-policy`, etc.
- 🧠 Built with [Optuna](https://optuna.org) for hyperparameter optimization
- ⚙️ Configurable via [Hydra](https://hydra.cc)
- 📊 Visualize results with Optuna Dashboard

---

## 📦 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/gjgjos/llm-serving-optimizer.git
cd llm-serving-optimizer
pip install -r requirements.txt
````

### 2. Serve Your Model with SGLang

```bash
docker run --gpus all -p 30000:30000 \
  -v /path/to/model:/model --ipc=host \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
    --model-path /model --host 0.0.0.0 --port 30000 \
    --served-model-name your-model-name
```

### 3. Run Optimization

```bash
python optimize.py
```

Outputs (latency, best config, db.sqlite3) are saved to `outputs/YYYY-MM-DD/HH-MM-SS/`.

---

## ⚙️ Configuration (Hydra)

Edit `config/config.yaml` to change:

* Model path, base URL
* Prompt lengths, concurrency
* Search space for Optuna

---

## 📊 Visualize with Dashboard

```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///outputs/.../db.sqlite3
```

Or via Docker:

```bash
docker run -it --rm -p 8080:8080 -v `pwd`:/app -w /app \
  ghcr.io/optuna/optuna-dashboard sqlite:///db.sqlite3
```

---

## ✅ Example Result

* ⏱ Latency reduced from **13,450ms → 10,128ms**
* Best config:

  * `chunked_prefill_size: 240`
  * `max_prefill_tokens: 3072`
  * `schedule_policy: fcfs`
  * `schedule_conservativeness: 0.8`

---

## 📎 Reference

* 📘 [Medium Article](https://medium.com/@yourusername/optimizing-llm-serving-speed-with-sglang-optuna-hydra-xxxxxxxx)