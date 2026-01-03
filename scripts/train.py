#!/usr/bin/env python3
"""
Simple local RL training script using REINFORCE.

Usage:
    uv run scripts/train.py --max-steps 100
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from binoculars import Binoculars
from binoculars.detector import BINOCULARS_ACCURACY_THRESHOLD


def main():
    parser = argparse.ArgumentParser(description="Train model to evade Binoculars detection")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B", help="Policy model")
    parser.add_argument("--max-steps", type=int, default=100, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-samples", type=int, default=1000, help="Dataset samples")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max tokens to generate")
    args = parser.parse_args()

    print("=" * 60)
    print("BINOCULARS RL TRAINING (REINFORCE)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer and model
    print(f"\nLoading policy model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.train()

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

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"\nStarting training for {args.max_steps} steps...")
    print("-" * 60)

    total_rewards = []

    for step in range(args.max_steps):
        # Sample batch
        indices = torch.randint(0, len(ds), (args.batch_size,)).tolist()
        batch_texts = [ds[i]["text"][:150] for i in indices]

        batch_rewards = []
        batch_loss = 0.0

        for text in batch_texts:
            # Create prompt
            prompt = f"Continue this story: {text}"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

            # Generate with gradients
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            generated_ids = outputs.sequences[0, inputs.input_ids.shape[1]:]
            completion = tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Compute reward
            if completion.strip():
                score = bino.compute_score(completion)
                reward = max(-2.0, min(2.0, score - threshold))
            else:
                reward = -2.0

            batch_rewards.append(reward)

            # Compute log probs for REINFORCE
            with torch.enable_grad():
                full_ids = outputs.sequences
                logits = model(full_ids).logits

                # Get log probs of generated tokens
                gen_logits = logits[0, inputs.input_ids.shape[1]-1:-1, :]
                gen_probs = F.log_softmax(gen_logits, dim=-1)
                token_log_probs = gen_probs.gather(1, generated_ids.unsqueeze(1)).squeeze(1)

                # REINFORCE loss: -reward * log_prob
                loss = -reward * token_log_probs.mean()
                batch_loss += loss

        # Update
        optimizer.zero_grad()
        (batch_loss / args.batch_size).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        mean_reward = sum(batch_rewards) / len(batch_rewards)
        total_rewards.append(mean_reward)

        if (step + 1) % 10 == 0:
            avg_reward = sum(total_rewards[-10:]) / min(10, len(total_rewards))
            print(f"Step {step + 1}/{args.max_steps} | Reward: {mean_reward:+.4f} | Avg(10): {avg_reward:+.4f}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final avg reward: {sum(total_rewards[-10:]) / 10:+.4f}")
    print("=" * 60)

    # Save model
    output_dir = Path("outputs/binoculars-rl")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel saved to: {output_dir}")


if __name__ == "__main__":
    main()
