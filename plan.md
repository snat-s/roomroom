# Binoculars RL Environment Plan

## Project Overview

Create an RL environment where a language model learns to write text that
passes as human-written according to the Binoculars AI detection method. The
model receives rewards based on Binoculars scores.

## Architecture

```
┌─────────────────┐     prompt      ┌─────────────────┐
│   TinyStories   │ ──────────────▶ │  Policy Model   │
│     Dataset     │                 │ (Qwen3-0.6B)    │
└─────────────────┘                 └────────┬────────┘
                                             │ completion
                                             ▼
                                    ┌─────────────────┐
                                    │   Binoculars    │
                                    │  (Qwen2.5-0.5B  │
                                    │   + Instruct)   │
                                    └────────┬────────┘
                                             │ score
                                             ▼
                                    ┌─────────────────┐
                                    │  Reward: score  │
                                    │  - threshold    │
                                    │  (clipped)      │
                                    └─────────────────┘
```

## Components

### 1. Dataset Loader
- **Source**: `roneneldan/TinyStories` from HuggingFace
- **Field**: Use the story prompts/beginnings as prompts
- **Processing**: Truncate to reasonable prompt length, filter empty

### 2. Binoculars Reward Function
- **Observer**: `Qwen/Qwen2.5-0.5B`
- **Performer**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Reward formula**: `clip(score - threshold, -2, 2)`
- **Threshold**: Start with 1.0 (may need calibration)

### 3. RL Environment (using `verifiers` framework)
- **Base class**: `SingleTurnEnv` (single prompt → completion)
- **Rubric**: Custom rubric with Binoculars-based reward function

## File Structure

```
roomroom/
├── binoculars/              # Existing detector code
│   ├── __init__.py
│   ├── detector.py
│   ├── metrics.py
│   └── utils.py
├── environments/
│   └── binoculars_env/
│       ├── __init__.py      # Environment implementation
│       └── README.md        # Environment docs
├── main.py                  # Entry point / demo
├── pyproject.toml           # Add verifiers dependency
└── scripts/
    └── calibrate.py         
```

## Implementation Steps

### Step 1: Add Dependencies DONE
We have added them.
Add to `pyproject.toml`:
- `verifiers` (from Prime Intellect)
- `datasets` (for TinyStories loading)

### Step 2: Create Environment Module
File: `environments/binoculars_env/__init__.py`

```python
import verifiers as vf
from binoculars import Binoculars
from datasets import load_dataset

THRESHOLD = 1.0  # May need calibration for Qwen models

def binoculars_reward(completion: str, binoculars: Binoculars, **kwargs) -> float:
    """Reward based on Binoculars score.

    Higher score = more human-like = positive reward
    Lower score = more AI-like = negative reward
    """
    score = binoculars.compute_score(completion)
    reward = score - THRESHOLD
    return max(-2.0, min(2.0, reward))  # clip to [-2, 2]

def load_environment(**kwargs):
    # Load TinyStories dataset
    # Dataset has 'text' field with stories
    ds = load_dataset("roneneldan/TinyStories", split="train")

    # Rename 'text' to 'prompt' for verifiers compatibility
    ds = ds.rename_column("text", "prompt")

    # Take subset for faster iteration
    ds = ds.select(range(min(10000, len(ds))))

    # Initialize Binoculars detector (shared across episodes)
    bino = Binoculars(
        observer_name_or_path="Qwen/Qwen2.5-0.5B",
        performer_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
    )

    # Create rubric with Binoculars as shared object
    rubric = vf.Rubric(
        funcs=[binoculars_reward],
        weights=[1.0],
        class_objects={"binoculars": bino}
    )

    # Create environment
    env = vf.SingleTurnEnv(
        dataset=ds,
        rubric=rubric,
        **kwargs
    )
    return env
```

### Step 3: Update main.py
Simple entry point to test the environment:
```python
from environments.binoculars_env import load_environment

env = load_environment()
# Run a sample rollout
```

### Step 4: Optional Calibration Script
DONE: We did something different, we used the Ghosbuster datasets with
500 samples to calibrate the models.

we specifically ran the following:

```bash
uv run scripts/calibrate.py --max-samples=500 --batch-size=1 --output=results/calibration_data.json
```
Create `scripts/calibrate.py` to:
1. Run Binoculars on known human text (TinyStories samples)
2. Run on known AI text (generate with Qwen)
3. Find optimal threshold that separates them

## Key Considerations

### Memory Management
- Binoculars loads 2 models (observer + performer)
- Policy model is a 3rd model
- Total: ~3 small models in VRAM


### Potential Failure Modes
1. **Adversarial overfitting**: Model learns tricks that fool Binoculars but don't make text more human-like
2. **Mode collapse**: Model generates repetitive "safe" outputs
3. **Reward hacking**: Model finds degenerate solutions (very short/long text)

**Mitigations**:
- Add length penalty/bonus
- Use KL divergence penalty against base model
- Monitor text quality during training

## Commands

```bash
# Install dependencies
uv sync

# Run evaluation to see how model currently performs
uv run python main.py

# Run calibration (to set thresholds)
uv run scripts/calibrate.py --max-samples=500 --batch-size=1 --output=results/calibration_data.json
```

## Training

### Option 1: Simple Local Training (TRL)

For single-GPU training using TRL's PPOTrainer:

```bash
uv run scripts/train.py --max-steps 100 --batch-size 4
```

### Option 2: Distributed Training (prime-rl)

For multi-GPU training with prime-rl:

1. Clone and setup prime-rl:
```bash
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl
uv sync --all-extras
```

2. Install our environment:
```bash
uv pip install -e /path/to/roomroom/environments/binoculars_env
```

3. Start tmux session:
```bash
bash scripts/tmux.sh
```

4. Run training (in separate panes):
```bash
# Inference pane
uv run inference @ /path/to/roomroom/configs/binoculars/infer.toml

# Trainer pane
uv run trainer @ /path/to/roomroom/configs/binoculars/train.toml

# Orchestrator pane
uv run orchestrator @ /path/to/roomroom/configs/binoculars/orch.toml
```

Config files are in `configs/binoculars/`:
- `infer.toml` - vLLM inference server
- `orch.toml` - Rollout generation + reward computation
- `train.toml` - PPO training
