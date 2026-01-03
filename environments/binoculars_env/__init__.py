"""
Binoculars RL Environment.

An environment where a model learns to write text that passes as human-written
according to the Binoculars AI detection method.
"""

import json
from pathlib import Path

import verifiers as vf
from datasets import load_dataset

from binoculars import Binoculars

CALIBRATION_FILE = Path(__file__).parent.parent.parent / "results" / "calibration_data.json"

# Default models and threshold
DEFAULT_OBSERVER = "Qwen/Qwen2.5-0.5B"
DEFAULT_PERFORMER = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_THRESHOLD = 1.0


def load_calibration():
    """Load calibrated threshold from file."""
    if CALIBRATION_FILE.exists():
        with open(CALIBRATION_FILE) as f:
            data = json.load(f)
        return data.get("accuracy_threshold", DEFAULT_THRESHOLD)
    return DEFAULT_THRESHOLD


def make_binoculars_reward(bino: Binoculars, threshold: float):
    """Create a reward function with captured Binoculars instance."""

    def binoculars_reward(completion: str, **kwargs) -> float:
        """
        Reward based on Binoculars score.

        Higher score = more human-like = positive reward
        Lower score = more AI-like = negative reward
        """
        if not completion or not completion.strip():
            return -2.0

        score = bino.compute_score(completion)
        reward = score - threshold
        return max(-2.0, min(2.0, reward))

    return binoculars_reward


def load_environment(
    observer: str = DEFAULT_OBSERVER,
    performer: str = DEFAULT_PERFORMER,
    threshold: float | None = None,
    max_samples: int = 10000,
    **kwargs,
):
    """
    Load the Binoculars RL environment.

    Args:
        observer: Observer model for Binoculars
        performer: Performer model for Binoculars
        threshold: Score threshold (loads from calibration if None)
        max_samples: Max dataset samples to use
        **kwargs: Passed to SingleTurnEnv

    Returns:
        vf.SingleTurnEnv configured with Binoculars reward
    """
    if threshold is None:
        threshold = load_calibration()

    print(f"Loading Binoculars environment...")
    print(f"  Observer: {observer}")
    print(f"  Performer: {performer}")
    print(f"  Threshold: {threshold:.4f}")

    # Load TinyStories dataset
    ds = load_dataset("roneneldan/TinyStories", split="train")
    ds = ds.rename_column("text", "prompt")
    ds = ds.select(range(min(max_samples, len(ds))))
    print(f"  Dataset: {len(ds)} samples from TinyStories")

    # Initialize Binoculars detector
    bino = Binoculars(
        observer_name_or_path=observer,
        performer_name_or_path=performer,
        use_bfloat16=True,
    )

    # Create rubric with closure-captured Binoculars
    reward_fn = make_binoculars_reward(bino, threshold)
    rubric = vf.Rubric(funcs=[reward_fn], weights=[1.0])

    # Create environment
    env = vf.SingleTurnEnv(dataset=ds, rubric=rubric, **kwargs)

    return env
