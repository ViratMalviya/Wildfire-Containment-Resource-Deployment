# 🤝 Contributing — Wildfire Containment & Resource Deployment

## Branching Strategy (Git Flow)

```
main            ─── stable, production-ready releases
  └── develop   ─── integration branch for features
       ├── feature/<name>
       ├── bugfix/<name>
       └── experiment/<name>
```

| Branch Type | Pattern | Example |
|---|---|---|
| Feature | `feature/<desc>` | `feature/mlflow-tracking` |
| Bugfix | `bugfix/<desc>` | `bugfix/reward-calculation` |
| Experiment | `experiment/<name>` | `experiment/dqn-agent` |
| Release | `release/v<ver>` | `release/v1.0.0` |

## Workflow

1. Create feature branch from `develop`
2. Commit with conventional messages: `feat(agent): add Boltzmann exploration`
3. Push and open Pull Request to `develop`
4. Code review (1 approval required)
5. Squash-merge into `develop`
6. Release: merge `develop` → `main`, tag with `git tag -a v1.0.0`

## Commit Types

| Type | Use |
|---|---|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation |
| `ci` | CI/CD changes |
| `experiment` | New ML experiment |

## Experiment Workflow

1. Create YAML config in `configs/`
2. `python train.py --config configs/<new>.yaml`
3. `python evaluate.py --config configs/<new>.yaml`
4. `git tag exp-<name>` and push

## Rollback Procedures

```bash
# Model rollback — use earlier checkpoint
python evaluate.py --policy models/policy_exp-qlearning-1_ep400.pkl

# Code rollback
git checkout exp-qlearning-1    # checkout experiment tag
git revert HEAD                 # revert last commit on main

# MLflow rollback — transition model stage via UI
# Production → Archived, Staging → Production
```

## Code Review Checklist

- [ ] PEP-8 compliant
- [ ] Docstrings on all functions
- [ ] YAML config valid
- [ ] Training converges (check curves)
- [ ] Results CSV/JSON generated
- [ ] requirements.txt updated if needed
