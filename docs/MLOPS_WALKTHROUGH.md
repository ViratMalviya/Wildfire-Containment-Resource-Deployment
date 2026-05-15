# 🚀 Deep-Dive: MLOps in the Wildfire Containment Project

Welcome to the MLOps Walkthrough for the **Wildfire Containment & Resource Deployment** project. This document serves as an educational guide to explain exactly **what** MLOps tools we are using, **why** they are necessary for an "Excellent" grade, and **how** they are applied specifically to our Reinforcement Learning (RL) wildfire agent.

---

## 1. The Core Problem: Why do we need MLOps?

In a standard university project, you might write a `train.py` script, run it a few times, save the best model as a `.pkl` file, and call it a day. But what happens when:
- You want to compare the results of a training run today with one you did last week?
- Your teammate wants to run the exact same code but their Python environment is different, so it crashes?
- You need to deploy this RL agent to a real-world firefighting dashboard, but you don't know how to package it?

**MLOps (Machine Learning Operations)** solves these problems. It takes a raw machine learning script and turns it into a **scalable, reproducible, and trackable software product**.

---

## 2. Experiment Tracking & Model Registry: **MLflow**

### The Problem
When training our Q-Learning agent, we tweak hyperparameters: learning rate ($\alpha$), discount factor ($\gamma$), and exploration rate ($\epsilon$). If we run 20 experiments, how do we remember which settings resulted in the agent that saved the most forest cells?

### The Solution: MLflow (`train.py` & `evaluate.py`)
MLflow is an industry-standard open-source platform for managing the ML lifecycle.

- **How it's used here:** Inside our scripts, we wrap our training loop with `mlflow.start_run()`. 
- **What it tracks:**
  - **Parameters:** We log `learning_rate`, `grid_size`, `wind_direction`, etc. using `mlflow.log_params()`.
  - **Metrics:** After every episode, we log `reward` and `total_burned`.
  - **Artifacts:** When training is done, we save the `Q-table` as a `.pkl` file and our reward curve plots (`.png`). MLflow automatically uploads these files to its server using `mlflow.log_artifact()`.
- **Model Registry:** The best performing policy is tagged and registered as `wildfire-qlearning-policy`. If we deploy a firefighting API, the API can ask MLflow: *"Give me version 1 of the wildfire-qlearning-policy."*

---

## 3. Data & Pipeline Versioning: **DVC (Data Version Control)**

### The Problem
Git is amazing for code, but it is terrible for large files (like big datasets or massive neural network models). If we commit 500MB `.pkl` models to Git every day, the repository will become unusable.

### The Solution: DVC (`dvc.yaml` & `dvc.lock`)
DVC acts exactly like Git, but specifically for data and models.

- **How it's used here:** We use DVC to define our **Machine Learning Pipeline**. In `dvc.yaml`, we define stages:
  1. `generate_data`
  2. `clean_data`
  3. `train_agent`
  4. `evaluate_agent`
- **Why it's powerful:** If you change the code in the `evaluate_agent` step but don't touch the training code, running `dvc repro` is smart enough to **only run the evaluation step**, caching the training step. It also creates a `dvc.lock` file that mathematically hashes the exact state of your data/models, meaning your teammate can download the *exact* Q-table you trained by typing `dvc pull`.

---

## 4. Continuous Integration / Continuous Deployment (CI/CD): **GitHub Actions**

### The Problem
How do we guarantee that when a teammate pushes new code to GitHub, it doesn't break the RL agent or introduce syntax errors? 

### The Solution: GitHub Actions (`.github/workflows/ml_pipeline.yml`)
GitHub Actions acts as a robot that automatically tests our code in the cloud every time someone pushes to the repository.

- **How it's used here:** Whenever code is pushed to `main` or `develop`, our pipeline wakes up a virtual Ubuntu server and does the following:
  1. **Linting (`flake8`):** Scans the Python code for PEP-8 formatting errors (like missing spaces or unused imports).
  2. **Unit Tests (`pytest`):** Runs tests to ensure the Wildfire Simulator logic hasn't broken.
  3. **Automated Training:** It literally runs `python train.py` for both of our configurations to ensure the model still compiles and learns.
  4. **Validation:** It evaluates the model and uploads the new comparison charts as a downloadable zip file on GitHub.

This ensures the `main` branch is **always in a working state**.

---

## 5. Containerization: **Docker**

### The Problem
"It works on my machine!" — The classic developer excuse. Your teammate might be on Windows with Python 3.11, while the production server is on Linux with Python 3.9. 

### The Solution: Docker (`Dockerfile` & `docker-compose.yml`)
Docker wraps our entire application (Python, libraries, scripts, and OS-level dependencies) into an isolated, standardized box called a **Container**.

- **How it's used here:** 
  - The `Dockerfile` provides the recipe: *"Start with Ubuntu, install Python 3.10, copy the codebase, run `pip install requirements.txt`."*
  - `docker-compose.yml` allows us to run multiple containers at once. With one command (`docker-compose up`), we spin up:
    1. Our **FastAPI Backend** (to serve model predictions)
    2. An **MLflow Tracking Server** (to view our experiments)
    3. A **Prometheus Server** (for monitoring)
- **Why it's necessary:** We can take this Docker container and drop it onto AWS, Google Cloud, or Azure, and it is mathematically guaranteed to run exactly the same way it runs on your laptop.

---

## 6. GitOps, Orchestration & Monitoring: **Kubernetes & Prometheus**

### The Problem
If this was a real application deployed by the California Department of Forestry, one Docker container running on a laptop isn't enough. What if millions of people check the dashboard? What if the server crashes? How do we know if the model is failing in the real world?

### The Solution: Kubernetes (`k8s/`), Helm (`helm/`), and Prometheus (`infra/`)
This is the pinnacle of enterprise MLOps.

- **Kubernetes (K8s):** An orchestrator that manages thousands of Docker containers. If the Wildfire API crashes, Kubernetes instantly restarts it. If traffic spikes, it creates 10 clones of the API to handle the load (Auto-scaling).
- **GitOps (ArgoCD):** We use a philosophy called GitOps. Instead of manually deploying to a server, we tell ArgoCD: *"Watch the `main` branch. Whenever `deployment.yaml` changes, automatically update the live server."* This means Git is the single source of truth for our infrastructure.
- **Monitoring (Prometheus):** Our FastAPI server exposes a `/metrics` endpoint. Prometheus scrapes this endpoint every 5 seconds. If the RL Agent starts suggesting terrible deployments (Model Drift) or if the server response time slows down, Prometheus triggers an alert to the engineers.

---

## Summary of the Journey

By integrating these tools, we transformed a basic Reinforcement Learning homework assignment into a **production-ready architecture**:

1. You write code and push to GitHub.
2. **GitHub Actions** automatically lints and tests the code.
3. The pipeline trains the agent, tracked meticulously by **MLflow**, and versions the data/models using **DVC**.
4. The approved model is wrapped in a **Docker** container.
5. **ArgoCD** sees the new container and deplops it to **Kubernetes**.
6. **Prometheus** monitors the live deployment for failures or model drift.

This represents the exact workflow utilized by top-tier tech companies, achieving the "Excellent" standard for your university rubric.
