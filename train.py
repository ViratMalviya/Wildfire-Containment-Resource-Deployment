"""
train.py — Main Training Script
=================================
Trains the Q-learning agent on the Wildfire Containment simulator.

Usage:
  python train.py --config configs/qlearning_v1.yaml

Features:
  - Loads config from YAML
  - Trains for N episodes with epsilon-greedy exploration
  - Logs per-episode metrics (reward, burned cells, epsilon)
  - Saves policy checkpoints at specified episodes
  - Writes experiment results to results/results_<experiment>.csv
  - Writes a log.json summary for MLOps tracking
  - MLflow integration: logs params, metrics, artifacts, and registers model

Reproducibility:
  "Run python train.py --config configs/qlearning_v1.yaml to get the same result."
"""

import argparse
import os
import sys
import json
import csv
import yaml
import numpy as np
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim.wildfire_env import WildfireEnv  # noqa: E402
from src.agent import QLearningAgent  # noqa: E402

# ── MLflow Integration ──
# Graceful fallback if MLflow is not installed
try:
    import mlflow
    import mlflow.pyfunc
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("[WARN] MLflow not installed — experiment tracking will use local JSON/CSV only.")
    print("       Install with: pip install mlflow")

if MLFLOW_AVAILABLE:
    class QLearnPolicyModel(mlflow.pyfunc.PythonModel):
        """Wraps the Q-learning policy for MLflow Model Registry."""
        def load_context(self, context):
            import pickle
            with open(context.artifacts["policy"], "rb") as f:
                self.policy_data = pickle.load(f)

        def predict(self, context, model_input):
            return {"policy_loaded": True, "q_table_size": len(self.policy_data.get("q_table", {}))}


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train(config):
    """Run the training loop with MLflow experiment tracking."""
    exp_name = config["experiment_name"]
    env_cfg = config["env"]
    agent_cfg = config["agent"]
    train_cfg = config["training"]

    print("=" * 60)
    print("  WILDFIRE CONTAINMENT — RL TRAINING")
    print(f"  Experiment: {exp_name}")
    print("  Algorithm : Q-Learning (tabular, epsilon-greedy)")
    print(f"  MLflow    : {'[OK] Active' if MLFLOW_AVAILABLE else '[X] Not installed'}")
    print("=" * 60)

    # --- Initialise environment ---
    env = WildfireEnv(**env_cfg)

    # --- Initialise agent ---
    agent = QLearningAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        learning_rate=agent_cfg["learning_rate"],
        discount_factor=agent_cfg["discount_factor"],
        epsilon=agent_cfg["epsilon"],
        epsilon_min=agent_cfg["epsilon_min"],
        epsilon_decay=agent_cfg["epsilon_decay"],
    )

    episodes = train_cfg["episodes"]
    log_interval = train_cfg["log_interval"]
    save_at = train_cfg.get("save_policy_at", [])
    policy_dir = train_cfg.get("policy_dir", "models/")

    os.makedirs(policy_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # ── MLflow: Set up experiment ──
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("Wildfire-Containment-RL")

    # Use MLflow context manager for auto-logging
    mlflow_run = None
    if MLFLOW_AVAILABLE:
        mlflow_run = mlflow.start_run(run_name=exp_name)
        # Log all hyperparameters
        mlflow.log_params({
            "experiment_name": exp_name,
            "algorithm": "Q-Learning",
            "learning_rate": agent_cfg["learning_rate"],
            "discount_factor": agent_cfg["discount_factor"],
            "epsilon_start": agent_cfg["epsilon"],
            "epsilon_min": agent_cfg["epsilon_min"],
            "epsilon_decay": agent_cfg["epsilon_decay"],
            "grid_size": env_cfg["grid_size"],
            "num_resources": env_cfg["num_resources"],
            "base_spread_prob": env_cfg["base_spread_prob"],
            "wind_direction": env_cfg["wind_direction"],
            "wind_spread_bonus": env_cfg["wind_spread_bonus"],
            "max_steps": env_cfg["max_steps"],
            "num_initial_fires": env_cfg["num_initial_fires"],
            "tree_density": env_cfg["tree_density"],
            "episodes": episodes,
        })
        # Log the config file as artifact
        mlflow.log_artifact(args.config, artifact_path="configs")

    # --- Training metrics storage ---
    all_rewards = []
    all_burned = []
    all_epsilons = []
    episode_logs = []

    print(f"\nTraining for {episodes} episodes...\n")
    start_time = time.time()

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        step = 0

        while True:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step += 1

            if done:
                break

        agent.decay_epsilon()

        total_burned = env.total_burned
        all_rewards.append(total_reward)
        all_burned.append(total_burned)
        all_epsilons.append(agent.epsilon)

        episode_logs.append({
            "episode": ep,
            "reward": round(total_reward, 2),
            "total_burned": int(total_burned),
            "epsilon": round(agent.epsilon, 4),
            "steps": step,
            "q_table_size": agent.get_q_table_size(),
        })

        # ── MLflow: Log per-episode metrics ──
        if MLFLOW_AVAILABLE:
            mlflow.log_metrics({
                "reward": round(total_reward, 2),
                "total_burned": int(total_burned),
                "epsilon": round(agent.epsilon, 4),
                "steps": step,
                "q_table_size": agent.get_q_table_size(),
            }, step=ep)

        # --- Periodic logging ---
        if ep % log_interval == 0 or ep == 1:
            avg_reward = np.mean(all_rewards[-log_interval:])
            avg_burned = np.mean(all_burned[-log_interval:])
            print(f"  Episode {ep:>4d}/{episodes}  |  "
                  f"Avg Reward: {avg_reward:>7.2f}  |  "
                  f"Avg Burned: {avg_burned:>5.1f}  |  "
                  f"Epsilon: {agent.epsilon:.4f}  |  "
                  f"Q-table: {agent.get_q_table_size()} states")

        # --- Save policy checkpoints ---
        if ep in save_at:
            policy_path = os.path.join(policy_dir,
                                       f"policy_{exp_name}_ep{ep}.pkl")
            agent.save(policy_path)
            if MLFLOW_AVAILABLE:
                mlflow.log_artifact(policy_path, artifact_path="checkpoints")

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.1f}s")

    # --- Save final policy ---
    final_path = os.path.join(policy_dir, f"policy_{exp_name}_final.pkl")
    agent.save(final_path)

    # --- Write results CSV ---
    csv_path = f"results/results_{exp_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "episode", "reward", "total_burned", "epsilon", "steps", "q_table_size"
        ])
        writer.writeheader()
        writer.writerows(episode_logs)
    print(f"[RESULTS] Episode log saved to {csv_path}")

    # --- Write log.json summary (MLOps tracking) ---
    log_summary = {
        "run_id": f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "experiment_name": exp_name,
        "timestamp": datetime.now().isoformat(),
        "episodes": episodes,
        "average_reward": round(float(np.mean(all_rewards)), 2),
        "average_burned": round(float(np.mean(all_burned)), 2),
        "final_reward_avg_50": round(float(np.mean(all_rewards[-50:])), 2),
        "final_burned_avg_50": round(float(np.mean(all_burned[-50:])), 2),
        "parameters": {
            "learning_rate": agent_cfg["learning_rate"],
            "discount_factor": agent_cfg["discount_factor"],
            "epsilon_start": agent_cfg["epsilon"],
            "epsilon_min": agent_cfg["epsilon_min"],
            "epsilon_decay": agent_cfg["epsilon_decay"],
            "grid_size": env_cfg["grid_size"],
            "wind_direction": env_cfg["wind_direction"],
            "base_spread_prob": env_cfg["base_spread_prob"],
        },
        "training_time_seconds": round(elapsed, 1),
        "q_table_states": agent.get_q_table_size(),
        "policy_files": [
            os.path.join(policy_dir, f"policy_{exp_name}_ep{ep}.pkl")
            for ep in save_at
        ] + [final_path],
    }

    # Add MLflow run ID if available
    if MLFLOW_AVAILABLE and mlflow_run:
        log_summary["mlflow_run_id"] = mlflow_run.info.run_id

    log_path = f"results/log_{exp_name}.json"
    with open(log_path, "w") as f:
        json.dump(log_summary, f, indent=2)
    print(f"[MLOPS]   Run summary saved to {log_path}")

    # ── MLflow: Log final artifacts and summary metrics ──
    if MLFLOW_AVAILABLE:
        # Log summary metrics
        mlflow.log_metrics({
            "final_avg_reward": round(float(np.mean(all_rewards)), 2),
            "final_avg_burned": round(float(np.mean(all_burned)), 2),
            "final_reward_last50": round(float(np.mean(all_rewards[-50:])), 2),
            "final_burned_last50": round(float(np.mean(all_burned[-50:])), 2),
            "training_time_seconds": round(elapsed, 1),
            "final_q_table_size": agent.get_q_table_size(),
        })

        # Log result files as artifacts
        mlflow.log_artifact(csv_path, artifact_path="results")
        mlflow.log_artifact(log_path, artifact_path="results")
        mlflow.log_artifact(final_path, artifact_path="models")

        # ── Register model in MLflow Model Registry ──
        try:
            model_name = "wildfire-qlearning-policy"

            mlflow.pyfunc.log_model(
                artifact_path="policy_model",
                python_model=QLearnPolicyModel(),
                artifacts={"policy": final_path},
            )
            registered = mlflow.register_model(
                model_uri=f"runs:/{mlflow_run.info.run_id}/policy_model",
                name=model_name,
            )
            print(f"[MLFLOW]  Model registered: {model_name} v{registered.version}")
        except Exception as e:
            # Graceful fallback — model still logged as artifact
            print(f"[MLFLOW]  Model registry note: {e}")
            print("[MLFLOW]  Policy artifact still logged successfully.")

        mlflow.end_run()
        print(f"[MLFLOW]  Run logged: {mlflow_run.info.run_id}")

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Experiment       : {exp_name}")
    print(f"  Episodes         : {episodes}")
    print(f"  Avg Reward (all) : {np.mean(all_rewards):.2f}")
    print(f"  Avg Burned (all) : {np.mean(all_burned):.2f}")
    print(f"  Avg Reward (last50): {np.mean(all_rewards[-50:]):.2f}")
    print(f"  Avg Burned (last50): {np.mean(all_burned[-50:]):.2f}")
    print(f"  Q-table states   : {agent.get_q_table_size()}")
    print(f"  Final epsilon    : {agent.epsilon:.4f}")
    if MLFLOW_AVAILABLE and mlflow_run:
        print(f"  MLflow Run ID    : {mlflow_run.info.run_id}")
    print("=" * 60)

    return all_rewards, all_burned, agent


# Global args for MLflow artifact logging
args = None


def main():
    global args
    parser = argparse.ArgumentParser(
        description="Train Q-learning agent for Wildfire Containment")
    parser.add_argument("--config", type=str,
                        default="configs/qlearning_v1.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
