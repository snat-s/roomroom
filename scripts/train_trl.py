#!/usr/bin/env python3
"""
TRL-based PPO training script with KL penalty.

Usage:
    uv run scripts/train_trl.py --max-steps 100
"""

import argparse
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

sys.path.insert(0, str(Path(__file__).parent.parent))

from binoculars import Binoculars
from binoculars.detector import BINOCULARS_ACCURACY_THRESHOLD


def main():
    parser = argparse.ArgumentParser(description="Train model with TRL PPO to evade Binoculars")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B", help="Policy model")
    parser.add_argument("--max-steps", type=int, default=100, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--mini-batch-size", type=int, default=2, help="Mini batch size for PPO")
    parser.add_argument("--max-samples", type=int, default=1000, help="Dataset samples")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--kl-coef", type=float, default=0.2, help="KL penalty coefficient")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max tokens to generate")
    args = parser.parse_args()

    print("=" * 60)
    print("BINOCULARS TRL TRAINING (PPO)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Important for generation

    # Load model with value head
    print(f"Loading policy model with value head: {args.model}")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=None,
    ).to(device)

    # Load reference model (frozen, for KL computation)
    print("Loading reference model...")
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=None,
    ).to(device)

    # Load Binoculars detector
    print("\nLoading Binoculars detector...")
    bino = Binoculars(
        observer_name_or_path="Qwen/Qwen2.5-0.5B",
        performer_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        use_bfloat16=True,
    )
    threshold = BINOCULARS_ACCURACY_THRESHOLD
    print(f"  Threshold: {threshold:.4f}")

    # Load dataset
    print(f"\nLoading TinyStories dataset ({args.max_samples} samples)...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    ds = ds.select(range(min(args.max_samples, len(ds))))

    # PPO Config
    ppo_config = PPOConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        kl_penalty="kl",
        init_kl_coef=args.kl_coef,
        target_kl=6.0,
        log_with=None,
    )

    # PPO Trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    print(f"\nStarting training for {args.max_steps} steps...")
    print(f"  KL coef: {args.kl_coef}")
    print("-" * 60)

    total_rewards = []

    for step in range(args.max_steps):
        # Sample batch of prompts
        indices = torch.randint(0, len(ds), (args.batch_size,)).tolist()
        prompts = [f"Continue this story: {ds[i]['text'][:100]}" for i in indices]

        # Tokenize prompts
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        query_tensors = [inputs.input_ids[i] for i in range(args.batch_size)]

        # Generate responses
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": True,
            "temperature": 0.7,
            "pad_token_id": tokenizer.eos_token_id,
        }

        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)

        # Decode responses and compute rewards
        rewards = []
        for i, response in enumerate(response_tensors):
            # Get only the generated part (exclude query)
            generated_ids = response[len(query_tensors[i]):]
            completion = tokenizer.decode(generated_ids, skip_special_tokens=True)

            if completion.strip():
                score = bino.compute_score(completion)
                reward = max(-2.0, min(2.0, score - threshold))
            else:
                reward = -2.0

            rewards.append(torch.tensor(reward, device=device))

        # PPO step (handles KL penalty internally)
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        mean_reward = sum(r.item() for r in rewards) / len(rewards)
        total_rewards.append(mean_reward)

        if (step + 1) % 10 == 0:
            avg_reward = sum(total_rewards[-10:]) / min(10, len(total_rewards))
            kl = stats.get("objective/kl", 0)
            print(f"Step {step + 1}/{args.max_steps} | Reward: {mean_reward:+.4f} | Avg(10): {avg_reward:+.4f} | KL: {kl:.4f}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final avg reward: {sum(total_rewards[-10:]) / 10:+.4f}")
    print("=" * 60)

    # Save model
    output_dir = Path("outputs/binoculars-ppo")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel saved to: {output_dir}")


if __name__ == "__main__":
    main()
