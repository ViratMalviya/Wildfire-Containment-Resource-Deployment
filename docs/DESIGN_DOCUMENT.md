# 📋 Design Document — Wildfire Containment & Resource Deployment

> **Version:** 1.0  
> **Date:** 2026-05-15  
> **Project:** Wildfire Containment & Resource Deployment using Reinforcement Learning  
> **Course:** 6th Semester — Reinforcement Learning + MLOps  
> **SDG Alignment:** SDG 13 (Climate Action) + SDG 15 (Life on Land)

---

## 1. Problem Statement

Wildfires cause massive environmental destruction, releasing ~8 billion tonnes of CO₂ annually and devastating ecosystems. Current firefighting resource deployment relies on human intuition, which is suboptimal under rapidly changing fire conditions.

**Objective:** Build a Reinforcement Learning agent that learns to optimally deploy firefighting resources (ground crews, helicopters) on a grid-based wildfire simulator to **minimise total burned area**, supporting SDG 13 (Climate Action) and SDG 15 (Life on Land).

---

## 2. Stakeholder Analysis

| Stakeholder | Role | Interest | Impact |
|---|---|---|---|
| **Forest Fire Services** | End-user | Optimised resource deployment for faster containment | High — directly benefits from reduced burned area |
| **Environmental Agencies** | Regulatory | CO₂ emission reduction, ecosystem preservation | High — policy and compliance alignment |
| **Local Communities** | Affected population | Safety, property protection, air quality | High — reduced fire damage = safer communities |
| **Emergency Response Teams** | Operational user | Decision support for resource allocation | Medium — augments human decision-making |
| **Insurance Companies** | Financial | Reduced wildfire damage claims | Medium — lower payouts with better containment |
| **Climate Scientists** | Research | Data-driven wildfire behaviour modelling | Medium — validates RL approach for climate problems |
| **Government/Policy Makers** | Oversight | Budget allocation for firefighting resources | Medium — evidence-based funding decisions |
| **Wildlife Conservation Orgs** | Advocacy | Biodiversity and habitat preservation | Medium — fewer fires = more habitats saved |

---

## 3. Use Cases

### UC-01: Train RL Agent for Wildfire Containment
- **Actor:** Data Scientist / Researcher
- **Precondition:** Environment configured, YAML experiment config available
- **Flow:**
  1. User selects experiment config (e.g., `qlearning_v1.yaml`)
  2. System initialises wildfire simulator and Q-learning agent
  3. Agent trains for N episodes with ε-greedy exploration
  4. System logs per-episode metrics (reward, burned cells, epsilon)
  5. System saves policy checkpoints at configured milestones
  6. System writes experiment summary to JSON + CSV
- **Postcondition:** Trained policy saved, experiment metrics recorded
- **Alternative Flow:** If training diverges, user adjusts hyperparameters and re-trains

### UC-02: Evaluate Trained Policy vs Baseline
- **Actor:** Data Scientist / Researcher
- **Precondition:** At least one trained policy exists
- **Flow:**
  1. User runs evaluation with a specific policy and config
  2. System runs N episodes with random baseline
  3. System runs N episodes with trained RL policy (ε=0)
  4. System computes comparison metrics and generates plots
  5. System outputs SDG impact analysis
- **Postcondition:** Comparison CSV, plots, and evaluation JSON saved

### UC-03: Interactive Wildfire Simulation (Frontend)
- **Actor:** Stakeholder / Evaluator / Demo Audience
- **Precondition:** Frontend served (browser access)
- **Flow:**
  1. User opens dashboard in browser
  2. User selects agent mode (Random vs RL)
  3. User configures wind direction, speed, fire count
  4. User starts simulation and observes grid visualization
  5. System displays live metrics (burning, burned, trees, reward)
- **Postcondition:** Simulation completes, user compares outcomes

### UC-04: Compare Experiment Results
- **Actor:** Data Scientist
- **Precondition:** Multiple experiments have been run
- **Flow:**
  1. User examines training curves on dashboard
  2. User compares reward convergence across experiments
  3. User analyses hyperparameter impact on performance
- **Postcondition:** Insights derived for model selection

### UC-05: Deploy Model to Production (Design Intent)
- **Actor:** MLOps Engineer
- **Precondition:** Best model registered in MLflow
- **Flow:**
  1. CI/CD pipeline validates code and runs tests
  2. MLflow model registry promotes model to "Production" stage
  3. Docker container deploys model for inference
  4. Monitoring tracks burned-area KPI and model drift
- **Postcondition:** Production model serving predictions

---

## 4. Functional Requirements

| FR-ID | Requirement | Priority | Status |
|---|---|---|---|
| FR-01 | The system shall simulate wildfire spread on a 2D grid with probabilistic propagation | Must Have | ✅ Implemented |
| FR-02 | The system shall model wind direction as a bias on fire spread probability | Must Have | ✅ Implemented |
| FR-03 | The system shall provide a Q-learning agent with ε-greedy exploration | Must Have | ✅ Implemented |
| FR-04 | The agent shall learn to deploy resources to minimise total burned area | Must Have | ✅ Implemented |
| FR-05 | The system shall save policy checkpoints at configurable episode milestones | Must Have | ✅ Implemented |
| FR-06 | The system shall log per-episode metrics to CSV and JSON | Must Have | ✅ Implemented |
| FR-07 | The system shall support multiple experiment configurations via YAML | Must Have | ✅ Implemented |
| FR-08 | The system shall evaluate trained policy against random baseline | Must Have | ✅ Implemented |
| FR-09 | The system shall generate training curve plots (reward, burned area) | Must Have | ✅ Implemented |
| FR-10 | The system shall provide an interactive web-based simulator dashboard | Should Have | ✅ Implemented |
| FR-11 | The system shall track experiments with MLflow (params, metrics, artifacts) | Must Have | ✅ Implemented |
| FR-12 | The system shall register models in MLflow Model Registry | Should Have | ✅ Implemented |

---

## 5. Non-Functional Requirements

| NFR-ID | Requirement | Category | Metric |
|---|---|---|---|
| NFR-01 | Training for 800 episodes must complete in < 30 seconds | Performance | Measured: ~5s |
| NFR-02 | The Q-table must handle ≥ 6000 states without memory issues | Scalability | Measured: ~1400 states, capacity for 6561 |
| NFR-03 | All experiments must be reproducible from config + code | Reproducibility | YAML configs + git tags |
| NFR-04 | The system must run on Python 3.8+ with minimal dependencies | Portability | 4 dependencies (numpy, pyyaml, matplotlib, mlflow) |
| NFR-05 | Frontend dashboard must load in < 2 seconds on modern browsers | Usability | Canvas-based rendering, no heavy frameworks |
| NFR-06 | Code must follow Python PEP-8 style guidelines | Maintainability | Consistent formatting throughout |
| NFR-07 | All core modules must have docstrings and inline comments | Maintainability | All files documented |
| NFR-08 | The system must be containerizable via Docker | Deployability | Dockerfile + docker-compose provided |
| NFR-09 | CI/CD pipeline must run on every push to main | Reliability | GitHub Actions configured |

---

## 6. Feasibility Analysis

### 6.1 Technical Feasibility
- **Algorithm:** Q-learning is well-suited for the discretised 10×10 grid (state space ~6561). Tabular methods converge reliably for this problem size.
- **Compute:** Training completes in ~5 seconds on a standard laptop — no GPU required.
- **Dependencies:** Only 4 Python packages (numpy, pyyaml, matplotlib, mlflow) — all well-maintained and cross-platform.
- **Frontend:** Vanilla HTML/CSS/JS dashboard requires no build tools, runs in any browser.

### 6.2 Operational Feasibility
- **Reproducibility:** YAML configs + git tags + requirements.txt ensure any team member can reproduce results.
- **MLOps:** MLflow provides industry-standard experiment tracking, model versioning, and registry.
- **Monitoring:** KPIs defined for production deployment (burned area, resource utilisation, model drift).

### 6.3 Economic Feasibility
- **Development Cost:** Open-source stack (Python, MLflow, GitHub Actions) — zero licensing cost.
- **Infrastructure:** Runs on local hardware; Docker enables cloud deployment when needed.
- **ROI Potential:** Even a 10% improvement in wildfire containment efficiency could save millions in firefighting costs.

---

## 7. Constraints

| Constraint | Impact | Mitigation |
|---|---|---|
| **Tabular Q-learning** doesn't scale beyond ~20×20 grids | Limits real-world applicability | Future: upgrade to DQN for larger state spaces |
| **Simplified fire model** ignores terrain, humidity, vegetation types | Reduces simulation fidelity | Future: integrate real topographic/weather data |
| **Single-agent** control; real fires need multi-agent coordination | Doesn't capture real operational complexity | Future: multi-agent RL (MARL) |
| **No temporal prediction** — agent is reactive, not predictive | Agent can't anticipate fire spread | Future: incorporate predictive fire models |
| **Grid discretisation** loses spatial resolution | Coarse decisions at quadrant level | Future: cell-level action space with function approximation |

---

## 8. Trade-off Analysis

| Trade-off | Option A | Option B | Decision | Rationale |
|---|---|---|---|---|
| **Algorithm** | Q-Learning (tabular) | DQN (neural net) | Q-Learning | Grid is small enough for tabular; simpler to debug, faster training, easier to explain |
| **State space** | Raw grid (100 cells) | Binned sectors (8 values) | Binned sectors | Raw grid too sparse for tabular Q-learning; sector binning enables convergence |
| **Action space** | Per-cell deployment (100 actions) | Per-sector deployment (4 actions) | Per-sector | Reduces Q-table size dramatically while maintaining strategic choice |
| **Exploration** | Boltzmann | ε-greedy with decay | ε-greedy | Simpler, well-understood, sufficient for this problem |
| **Experiment tracking** | Custom JSON/CSV | MLflow | Both | JSON/CSV for quick inspection, MLflow for industry-standard tracking |
| **Frontend** | React/Next.js | Vanilla JS + Canvas | Vanilla JS | No build step, instant load, sufficient for demo dashboard |

---

## 9. Risk Assessment

| Risk-ID | Risk | Probability | Impact | Mitigation |
|---|---|---|---|---|
| R-01 | Q-learning fails to converge with more fire sources | Medium | High | Monitor training curves; increase episodes; tune hyperparameters |
| R-02 | Stochastic fire spread causes high variance in results | High | Medium | Evaluate over 100+ episodes; use averaged metrics |
| R-03 | Q-table becomes too large for complex scenarios | Low | High | Sector-based state compression limits growth to ~6561 states |
| R-04 | Policy overfits to specific wind direction | Medium | Medium | Train separate policies per wind config; document in evaluation |
| R-05 | MLflow server unavailable in production | Low | Medium | Fallback to local file tracking; Docker ensures isolation |
| R-06 | Model drift in real deployment (changing climate) | High (production) | High | Monitoring KPIs + automated retraining triggers (design plan) |

---

## 10. Traceability Matrix

| Requirement | Implementation | Test/Validation | Config |
|---|---|---|---|
| FR-01: Grid simulation | `sim/wildfire_env.py` → `WildfireEnv` class | `evaluate.py` runs 100 episodes | `env.grid_size`, `env.tree_density` |
| FR-02: Wind model | `wildfire_env.py` → `_build_wind_map()` | Compare N vs NE wind experiments | `env.wind_direction`, `env.wind_spread_bonus` |
| FR-03: Q-learning agent | `src/agent.py` → `QLearningAgent` class | Training convergence curve | `agent.learning_rate`, `agent.discount_factor` |
| FR-04: Resource deployment | `wildfire_env.py` → `step()` method | Baseline vs RL comparison | `env.num_resources` |
| FR-05: Policy checkpoints | `train.py` → lines 131-134 | Check `models/` directory | `training.save_policy_at` |
| FR-06: Metrics logging | `train.py` → CSV writer + JSON dump | Inspect `results/` files | `training.log_interval` |
| FR-07: YAML configs | `configs/qlearning_v1.yaml`, `qlearning_v2.yaml` | Reproducibility test | N/A |
| FR-08: Baseline evaluation | `evaluate.py` → `run_baseline()`, `run_rl_policy()` | Comparison CSV output | `--eval_episodes` |
| FR-09: Training plots | `evaluate.py` → `plot_training_curves()` | Check `results/*.png` | N/A |
| FR-10: Web dashboard | `frontend/index.html` + JS modules | Manual browser testing | N/A |
| FR-11: MLflow tracking | `train.py` → `mlflow.log_*` calls | MLflow UI inspection | `MLFLOW_TRACKING_URI` |
| FR-12: Model registry | `train.py` → `mlflow.register_model()` | MLflow Models page | `MLFLOW_TRACKING_URI` |

---

## 11. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER / DASHBOARD                      │
│              (frontend/index.html + JS)                  │
└──────────────────────┬──────────────────────────────────┘
                       │ Visualisation
┌──────────────────────▼──────────────────────────────────┐
│                 EXPERIMENT LAYER                         │
│  ┌──────────┐  ┌───────────┐  ┌──────────────────┐      │
│  │ train.py │  │evaluate.py│  │ configs/*.yaml   │      │
│  └────┬─────┘  └─────┬─────┘  └──────────────────┘      │
│       │              │                                   │
│  ┌────▼──────────────▼─────┐                             │
│  │    MLflow Tracking       │  ← params, metrics,        │
│  │    & Model Registry      │    artifacts, model versions│
│  └──────────────────────────┘                             │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                   CORE LAYER                             │
│  ┌─────────────────┐    ┌────────────────────────┐      │
│  │ sim/wildfire_env │    │  src/agent.py          │      │
│  │   .py            │◄──►│  (Q-Learning Agent)    │      │
│  │ (Environment)    │    │                        │      │
│  └─────────────────┘    └────────────────────────┘      │
└─────────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                  INFRA / MLOPS                           │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐     │
│  │ Docker   │  │ GitHub   │  │ DVC (data version) │     │
│  │ Compose  │  │ Actions  │  │                    │     │
│  └──────────┘  └──────────┘  └────────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

---

## 12. Glossary

| Term | Definition |
|---|---|
| **RL** | Reinforcement Learning — learning by trial-and-error with rewards |
| **Q-Learning** | Off-policy, tabular RL algorithm that learns action values Q(s,a) |
| **ε-greedy** | Exploration strategy: random action with probability ε, best action otherwise |
| **Firebreak** | A gap in vegetation that stops fire spread |
| **MLflow** | Open-source platform for ML experiment tracking and model management |
| **DVC** | Data Version Control — Git-like versioning for datasets and models |
| **CI/CD** | Continuous Integration / Continuous Deployment |
| **SDG** | United Nations Sustainable Development Goals |
| **KPI** | Key Performance Indicator |
