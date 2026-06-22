# AI Football Builder

Workspace for the "AI learns to play football" short series.

Planned steps:

- Build a simple football field simulation.
- Add an agent, ball physics, goals, and scoring.
- Train a shared self-play policy with PPO.
- Export replay milestones as a vertical Shorts video.
- Connect the finished builder to the upload pipeline.

## Commands

Preview the scripted environment:

```bash
python src/builders/ai_football_builder/video_generator.py
```

Run a tiny PPO smoke test:

```bash
python src/builders/ai_football_builder/rl_trainer_ppo.py smoke
```

Train the football policy:

```bash
python src/builders/ai_football_builder/rl_trainer_ppo.py train --updates 1000 --rollout-steps 1024
```

Training starts with a scripted behavior warm-start so PPO does not begin from
pure random wandering. Disable it with `--no-pretrain` if you want raw PPO only,
or tune it with `--pretrain-steps`.

The current trainer uses a fresh 35-value role-aware observation. Old checkpoints
from the earlier 25-value setup are intentionally rejected; retrain from scratch
after this rewrite.

Useful training log fields:

- `assistG`: goals scored shortly after a successful pass.
- `clump%`: percentage of rollout steps where teammates were too close.
- `mateD`: average closest teammate distance in pixels.

Preview the latest trained policy:

```bash
python src/builders/ai_football_builder/rl_trainer_ppo.py preview
```

Export a replay video:

```bash
python src/builders/ai_football_builder/rl_trainer_ppo.py export
```
