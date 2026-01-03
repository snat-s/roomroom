# AGENTS.md

This repository implements a Binoculars-based RL environment for training
language models to write more human-like text. The environment uses the
[Binoculars](https://github.com/ahans30/Binoculars) AI detection method as a
reward signal, built on the
[`verifiers`](https://github.com/primeintellect-ai/verifiers) framework.

## Project Overview

**Goal**: Train a language model (Qwen3-0.6B) to generate text that passes as human-written according to the Binoculars detector.

**Core Idea**: Use the Binoculars score as an RL reward signal:
- Higher scores → text appears more human-like → positive reward
- Lower scores → text appears AI-generated → negative reward

## Local Setup

```bash
# Clone and enter repo
git clone <repo-url>
cd roomroom

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync

# Verify installation
uv run python main.py
```

### GPU Requirements

The full setup loads 3 models:
| Model | Role | Size |
|-------|------|------|
| Qwen/Qwen2.5-0.5B | Binoculars observer | ~1GB |
| Qwen/Qwen2.5-0.5B-Instruct | Binoculars performer | ~1GB |
| Qwen/Qwen3-0.6B | Policy model (trainable) | ~1.2GB |

**Minimum**: 16GB VRAM (with offloading)
**Recommended**: 24GB VRAM

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

## Repository Structure

```
roomroom/
├── binoculars/                  # Binoculars detector (from paper)
│   ├── __init__.py              # Exports Binoculars class
│   ├── detector.py              # Main detector implementation
│   ├── metrics.py               # Perplexity and entropy functions
│   └── utils.py                 # Tokenizer consistency checks
├── environments/
│   └── binoculars_env/
│       ├── __init__.py          # Environment + load_environment()
│       └── README.md            # Environment-specific docs
├── scripts/
│   └── calibrate.py             # Threshold calibration utility
├── main.py                      # Entry point / demo
├── pyproject.toml               # Dependencies
├── plan.md                      # Implementation plan
└── AGENTS.md                    # This file
```

## How Binoculars Works

Binoculars detects AI-generated text by comparing two language models:
- **Observer**: Base model (Qwen2.5-0.5B)
- **Performer**: Instruction-tuned variant (Qwen2.5-0.5B-Instruct)

**Score calculation**:
```
score = perplexity(performer) / cross_entropy(observer, performer)
```

**Interpretation**:
- **Low score** (~0.5-0.8): Both models predict similarly → AI-generated
- **High score** (~1.0-1.5): Models disagree → Human-written

## Environment Implementation

### Reward Function

```python
THRESHOLD = 1.0  # Calibrate for your model pair

def binoculars_reward(completion: str, binoculars: Binoculars, **kwargs) -> float:
    score = binoculars.compute_score(completion)
    reward = score - THRESHOLD
    return max(-2.0, min(2.0, reward))  # clip to [-2, 2]
```

**Why centered continuous reward?**
1. Preserves gradient information (vs binary -1/+1)
2. Zero-centered works well with PPO
3. Clipping prevents outlier destabilization

### Dataset

Uses `roneneldan/TinyStories` - simple children's stories that represent natural human writing patterns.

```python
ds = load_dataset("roneneldan/TinyStories", split="train")
ds = ds.rename_column("text", "prompt")  # verifiers expects 'prompt'
```

## Coding Principles

### Style & Structure

- Format with `uv run ruff check --fix .`
- Use type annotations for public APIs
- Keep the binoculars module unchanged (reference implementation)
- New functionality goes in `environments/` or `scripts/`

### Error Handling

- Fail fast if models fail to load
- Validate GPU availability before training
- Log Binoculars scores during training for debugging

### Performance

- Binoculars scoring is the bottleneck (two forward passes)
- Batch completions when possible: `bino.compute_score([text1, text2, ...])`
- Consider gradient checkpointing for policy model if memory-constrained

## Extending the Environment

### Adding New Reward Components

Combine Binoculars with other rewards in the rubric:

```python
def length_penalty(completion: str, **kwargs) -> float:
    """Penalize very short or very long outputs."""
    length = len(completion.split())
    if length < 20:
        return -0.5
    if length > 500:
        return -0.5
    return 0.0

rubric = vf.Rubric(
    funcs=[binoculars_reward, length_penalty],
    weights=[1.0, 0.3],
    class_objects={"binoculars": bino}
)
```

### Using Different Model Pairs

Modify `load_environment()` to accept model arguments:

```python
def load_environment(
    observer: str = "Qwen/Qwen2.5-0.5B",
    performer: str = "Qwen/Qwen2.5-0.5B-Instruct",
    **kwargs
):
    bino = Binoculars(
        observer_name_or_path=observer,
        performer_name_or_path=performer,
    )
    # ...
```

### Calibrating Thresholds

The default threshold (1.0) is a starting point. To calibrate:

```bash
uv run python scripts/calibrate.py
```

This script:
1. Scores human text samples (TinyStories)
2. Scores AI-generated text (from policy model)
3. Finds threshold maximizing separation (F1 or low-FPR)

## Training

### With verifiers

```bash
# Evaluate environment
uv run vf-eval -s binoculars_env -m gpt-4.1 -n 10

# Full training (when ready)
uv run vf-train -s binoculars_env -m Qwen/Qwen3-0.6B
```

### Monitoring

Key metrics to track:
- **mean_reward**: Should increase over training
- **binoculars_score**: Raw scores before centering
- **completion_length**: Watch for length gaming
- **text_diversity**: Monitor for mode collapse

## Known Limitations

1. **Adversarial overfitting**: Model may learn artifacts that fool Binoculars without genuinely improving writing quality
2. **Threshold sensitivity**: Different model pairs need different thresholds
3. **Domain specificity**: Trained on TinyStories, may not generalize

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size
- Enable `use_bfloat16=True` (default)
- Use gradient checkpointing
- Offload Binoculars models to CPU during policy updates

### Low/Unstable Rewards
- Check threshold calibration
- Verify completions aren't empty/truncated
- Inspect raw Binoculars scores with `bino.compute_score()`

### Tokenizer Mismatch Error
Observer and performer must share the same tokenizer. Use models from the same family (e.g., both Qwen2.5).

## References

- [Binoculars Paper](https://arxiv.org/abs/2401.12070): Spotting LLM-Generated Text
- [verifiers Framework](https://github.com/primeintellect-ai/verifiers): RL environment framework
- [TinyStories Dataset](https://huggingface.co/datasets/roneneldan/TinyStories): Training data
