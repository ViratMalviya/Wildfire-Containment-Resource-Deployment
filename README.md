# 🔥 Wildfire Containment & Resource Deployment — RL + MLOps

> **RL Problem Statement:** An agent decides where to deploy firefighting resources
> (ground crews, helicopters) on a grid-based wildfire simulator to **minimise
> burned area**.

## 🌍 SDG Alignment

| SDG | Connection |
|-----|-----------|
| **SDG 13 — Climate Action** | Reducing wildfire damage lowers CO₂ emissions and helps combat climate change |
| **SDG 15 — Life on Land** | Preserving forest ecosystems protects terrestrial biodiversity and habitats |

---

## 📁 Project Structure

```
wildfire-rl/
├── .github/
│   ├── workflows/
│   │   └── ml_pipeline.yml       # CI/CD pipeline (lint → train → evaluate → docker)
│   └── PULL_REQUEST_TEMPLATE.md  # PR template for code reviews
├── sim/                           # Wildfire simulator
│   ├── __init__.py
│   └── wildfire_env.py            # Grid-based fire spread environment
├── src/                           # RL agent source
│   ├── __init__.py
│   └── agent.py                   # Q-Learning agent (tabular, ε-greedy)
├── configs/                       # Experiment configurations (YAML)
│   ├── qlearning_v1.yaml          # Conservative exploration
│   └── qlearning_v2.yaml          # More exploration + higher LR
├── models/                        # Saved policy checkpoints (.pkl)
├── results/                       # Experiment logs, CSVs, plots
├── experiments/                   # Experiment notes
├── docs/
│   └── DESIGN_DOCUMENT.md         # Full problem analysis & requirements
├── frontend/                      # Interactive dashboard
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   ├── simulator.js
│   ├── charts.js
│   └── data.js
├── train.py                       # Training script (MLflow-integrated)
├── evaluate.py                    # Baseline vs RL comparison + plots
├── Dockerfile                     # Container for reproducible execution
├── docker-compose.yml             # Full stack: MLflow + Train + Evaluate
├── dvc.yaml                       # DVC pipeline for data/model versioning
├── Makefile                       # Declarative automation commands
├── CONTRIBUTING.md                # Branching strategy & collaboration guide
├── requirements.txt
├── .gitignore
└── README.md                      # This file
```

---

## 🚀 Quick Start (Running Locally)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Agent

```bash
python train.py --config configs/qlearning_v1.yaml
python train.py --config configs/qlearning_v2.yaml
```

### 3. Evaluate Policies

```bash
python evaluate.py --config configs/qlearning_v1.yaml
```

### 4. Start the Application Stack

You can run the full system (Backend API and Frontend Dashboard) manually:

**Terminal 1 (Backend API):**
```bash
# Set encoding for Windows environments
export PYTHONIOENCODING=utf-8 
pip install fastapi uvicorn
python api/app.py
```
*API runs on `http://localhost:8000`*

**Terminal 2 (Frontend UI):**
```bash
python -m http.server 8080 -d frontend
```
*Dashboard runs on `http://localhost:8080`*

### 🛑 How to Stop Gracefully

If you are running the scripts manually in your terminal, press `Ctrl+C` in the respective terminal windows to safely terminate the running servers and python scripts.

---

## 🐳 Docker Deployment (Recommended)

The easiest way to run the **entire stack** (MLflow Tracking, Training, Evaluation, Backend API, and Frontend Dashboard) is using Docker.

### Start the Full Stack

```bash
docker-compose up -d --build
```

**Services Available:**
- 🖥️ **Frontend Dashboard**: `http://localhost:8080`
- ⚡ **Backend API Docs**: `http://localhost:8000/docs`
- 📊 **MLflow Tracking UI**: `http://localhost:5000`

### Stop the Full Stack Gracefully

To safely spin down the containers, stop the servers, and free up ports without losing your data:

```bash
docker-compose down
```

*(Note: Data is persisted in the `mlflow-data` volume and the mapped `results` directory).*

---

## 🎮 Simulator

The **WildfireEnv** is a grid-based wildfire spread simulator:

- **Grid:** 10×10 cells, each cell is one of: `EMPTY`, `TREE 🌲`, `BURNING 🔥`, `BURNED ⬛`, `FIREBREAK 🧱`
- **Fire Spread:** Probabilistic — each burning cell may ignite neighbouring trees, biased by wind direction
- **Resources:** The agent deploys firefighting resources to cells to create firebreaks or suppress active fires
- **Episode ends** when: fire is fully contained OR max steps reached

### State, Action, Reward

| Component | Description |
|-----------|------------|
| **State** | Binned (burning, trees) per quadrant — compact tuple for Q-table lookup |
| **Action** | Sector index 0–3 — which quadrant to deploy resources to |
| **Reward** | `−(newly burned cells) + 2.0 × (cells suppressed)` — penalises fire spread, rewards suppression |

---

## 🤖 RL Algorithm

### Algorithm Choice: **Q-Learning**

> *"Q-learning is chosen because the state space (discretised grid) is manageable
> for a tabular approach, and Q-learning's off-policy nature allows efficient
> learning from exploratory actions."*

### Exploration Strategy: **ε-greedy with decay**

- Start with ε = 1.0 (fully random)
- Decay by factor of 0.995 per episode
- Minimum ε = 0.05

### Q-Learning Update Rule

```
Q(s, a) ← Q(s, a) + α · [r + γ · max_a' Q(s', a') − Q(s, a)]
```

Where:
- α = learning rate (0.1)
- γ = discount factor (0.95)
- r = immediate reward

### Convergence

> *"Average reward improves over time and stabilises — the agent learns to
> strategically place firebreaks near the fire front rather than deploying
> resources randomly."*

---

## 📊 MLOps Pipeline

### Experiment Tracking (MLflow)

Every training and evaluation run is automatically logged to **MLflow**:

| What | How |
|------|-----|
| **Hyperparameters** | `mlflow.log_params()` — all agent & env config |
| **Per-episode metrics** | `mlflow.log_metrics()` — reward, burned, epsilon, steps |
| **Summary metrics** | Final avg reward, burned, training time |
| **Artifacts** | Config YAML, results CSV/JSON, policy checkpoints, plots |
| **Model Registry** | Final policy registered as `wildfire-qlearning-policy` |

### View Experiments

```bash
mlflow ui --port 5000
```

### Results CSV

Each training run produces `results/results_<experiment>.csv` containing:

| Column | Description |
|--------|------------|
| `episode` | Episode number |
| `reward` | Total episode reward |
| `total_burned` | Total cells burned |
| `epsilon` | Current exploration rate |
| `steps` | Steps taken in episode |
| `q_table_size` | Number of states discovered |

### Policy Versions (Model Registry)

| Policy File | Description |
|------------|------------|
| `policy_exp-qlearning-1_ep400.pkl` | V1 policy at 400 episodes (still exploring) |
| `policy_exp-qlearning-1_final.pkl` | V1 final policy (800 episodes, converged) |
| `policy_exp-qlearning-2_ep400.pkl` | V2 policy at 400 episodes |
| `policy_exp-qlearning-2_final.pkl` | V2 final policy |

---

## 🔁 CI/CD Pipeline

GitHub Actions runs automatically on every push to `main`/`develop`:

| Job | Description |
|-----|------------|
| **Lint** | flake8 + YAML validation |
| **Train** | Runs both experiments (matrix strategy) |
| **Evaluate** | Baseline vs RL comparison for both configs |
| **Docker** | Builds and validates container |

See [`.github/workflows/ml_pipeline.yml`](.github/workflows/ml_pipeline.yml)

---

## 🔄 Reproducibility & Versioning

### Reproduce a Run

```bash
git clone https://github.com/cheeseburden/Wildfire-Containment-Resource-Deployment.git
pip install -r requirements.txt
python train.py --config configs/qlearning_v1.yaml
python evaluate.py --config configs/qlearning_v1.yaml
```

### DVC Pipeline

```bash
dvc repro          # Run full pipeline
dvc metrics show   # Compare metrics across versions
dvc plots show     # View training curves
```

### Git Tags

```bash
git tag -l              # List experiment tags
git checkout exp-qlearning-1   # Checkout experiment 1 state
```

### Rollback

```bash
# Model rollback
python evaluate.py --policy models/policy_exp-qlearning-1_ep400.pkl

# Code rollback
git revert HEAD
```

### Make Commands

```bash
make install     # Install deps
make train       # Train both experiments
make evaluate    # Evaluate both
make all         # Full pipeline
make mlflow      # Start MLflow UI
make docker-up   # Start Docker stack
make clean       # Clean generated files
```

---

## 📈 Baseline vs RL Comparison

Evaluated over **100 episodes** per agent on the same environment. Results from two independent experiments with different hyperparameter configurations:

### Experiment 1 — Conservative (α=0.10, ε-decay=0.995, N wind, 2 fires)

| Metric | Random Baseline | RL Policy | Improvement |
|--------|---------------:|----------:|------------:|
| Avg Reward | −345.91 | −266.71 | **+23.1%** |
| Avg Burned Cells | 392.0 | 306.9 | **−21.7%** 🔥 |

### Experiment 2 — Aggressive (α=0.15, ε-decay=0.990, NE wind, 3 fires)

| Metric | Random Baseline | RL Policy | Improvement |
|--------|---------------:|----------:|------------:|
| Avg Reward | −370.21 | −350.69 | **+5.3%** |
| Avg Burned Cells | 417.8 | 398.8 | **−4.5%** |

> **Key finding:** Experiment 1 (slower ε-decay) outperforms Experiment 2 significantly.
> The agent needs sufficient exploration time before exploiting — faster decay converges
> prematurely to a suboptimal policy on a harder environment (NE wind, 3 fires).

### When RL Performs Better

- When fire starts in predictable locations — agent learns optimal firebreak placement
- When wind direction is consistent — agent exploits directional knowledge
- With enough training episodes for ε-greedy to converge (≥600 episodes)

### When RL May Struggle

- Highly stochastic fire spread (high base probability) — stochasticity dominates the signal
- Multiple simultaneous fire outbreaks from random start positions
- Very large grids — addressed by upgrading to the CNN DQN agent (`src/models/dqn_agent.py`)

---

## 📡 Monitoring Plan (Design Only — No Live Deployment)

If this system were deployed in real-world wildfire management, we would monitor:

- **Burned area per incident** — primary KPI; should decrease over time
- **Resource utilisation rate** — fraction of deployed resources actively suppressing fire
- **Response time** — time between fire detection and resource deployment
- **False positive rate** — resources deployed to non-threatened areas
- **Model drift** — degradation of policy performance as climate/vegetation patterns change
- **Safety constraints** — ensure no resources deployed into actively burning zones (crew safety)

---

## 🏷️ Git & GitOps

### Branching Strategy

```
main ─── stable releases
  └── develop ─── integration branch
       ├── feature/<name>
       ├── bugfix/<name>
       └── experiment/<name>
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for full workflow, commit conventions, and rollback procedures.

### Experiment Tags

```bash
git tag exp-qlearning-1   # After first experiment
git tag exp-qlearning-2   # After second experiment
```

---

## 🌱 SDG Impact

> *"Reducing average burned area by X% supports SDG 13 (Climate Action) by
> lowering CO₂ and particulate emissions from wildfires, and SDG 15 (Life on
> Land) by preserving forest ecosystems and protecting biodiversity."*

Optimised resource deployment also:
- Reduces firefighting costs and response times
- Minimises risk to human lives and infrastructure
- Supports data-driven disaster management strategies

---

## 📋 Results & Limitations

### Results
- Q-learning agent learns to strategically deploy resources near fire fronts
- Trained policy consistently outperforms random baseline
- Performance improves and stabilises after ~300 episodes

### Limitations
- Tabular Q-learning doesn't scale well to larger grids (20×20+)
- Simplified fire model doesn't capture terrain, humidity, vegetation types
- Single-agent control — real wildfire response involves multi-agent coordination
- No temporal fire prediction — agent is reactive, not predictive

### Future Work
- Deep Q-Network (DQN) for larger state spaces
- Multi-agent RL for coordinated resource deployment
- Integration with real satellite fire detection data

---

## 📝 Documentation

| Document | Description |
|----------|------------|
| [`docs/DESIGN_DOCUMENT.md`](docs/DESIGN_DOCUMENT.md) | Full problem analysis: stakeholders, use cases, requirements, feasibility, risks, traceability |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Branching strategy, commit conventions, rollback procedures, team workflow |
| [`.github/workflows/ml_pipeline.yml`](.github/workflows/ml_pipeline.yml) | CI/CD pipeline definition |
| [`dvc.yaml`](dvc.yaml) | DVC pipeline stages for data/model versioning |
