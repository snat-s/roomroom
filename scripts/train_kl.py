#!/usr/bin/env python3
"""
REINFORCE training with KL penalty to prevent mode collapse.

Usage:
    uv run scripts/train_kl.py --max-steps 100
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


def compute_kl_divergence(policy_logits, ref_logits):
    """Compute KL divergence between policy and reference model."""
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    ref_probs = F.softmax(ref_logits, dim=-1)
    kl = (ref_probs * (ref_probs.log() - policy_log_probs)).sum(dim=-1)
    return kl.mean()


def compute_perplexity(logits, target_ids):
    """Compute perplexity of target_ids under the model's logits."""
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, target_ids.unsqueeze(2)).squeeze(2)
    avg_neg_log_prob = -token_log_probs.mean()
    perplexity = torch.exp(avg_neg_log_prob)
    return perplexity


def main():
    parser = argparse.ArgumentParser(description="Train model with KL penalty")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B", help="Policy model")
    parser.add_argument("--max-steps", type=int, default=100, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-samples", type=int, default=1000, help="Dataset samples")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--kl-coef", type=float, default=0.1, help="KL penalty coefficient")
    parser.add_argument("--coherence-coef", type=float, default=0.0, help="Coherence penalty (ref model perplexity)")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max tokens to generate")
    args = parser.parse_args()

    print("=" * 60)
    print("BINOCULARS RL TRAINING (REINFORCE + KL)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer and policy model
    print(f"\nLoading policy model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    ).to(device)
    policy.train()
    for param in policy.parameters():
        param.requires_grad = True

    # Load reference model (frozen copy for KL)
    print("Loading reference model (frozen)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    ).to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Load Binoculars detector
    print("\nLoading Binoculars detector...")
    bino = Binoculars(
        observer_name_or_path="Qwen/Qwen2.5-0.5B",
        performer_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        use_bfloat16=True,
    )
    threshold = BINOCULARS_ACCURACY_THRESHOLD
    print(f"  Threshold: {threshold:.4f}")
    print(f"  KL coef: {args.kl_coef}")
    print(f"  Coherence coef: {args.coherence_coef}")

    # Load dataset
    print(f"\nLoading TinyStories dataset ({args.max_samples} samples)...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    ds = ds.select(range(min(args.max_samples, len(ds))))

    # Optimizer
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    print(f"\nStarting training for {args.max_steps} steps...")
    print("-" * 60)

    total_rewards = []
    total_kls = []

    for step in range(args.max_steps):
        # Sample batch
        indices = torch.randint(0, len(ds), (args.batch_size,)).tolist()
        batch_texts = [ds[i]["text"][:150] for i in indices]

        batch_rewards = []
        batch_kls = []
        batch_loss = torch.tensor(0.0, device=device, requires_grad=True)

        for text in batch_texts:
            # Create prompt
            prompt = f"Continue this story: {text}"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

            # Generate with policy (no grad for generation)
            with torch.no_grad():
                outputs = policy.generate(
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

            # Compute Binoculars reward
            if completion.strip():
                score = bino.compute_score(completion)
                reward = max(-2.0, min(2.0, score - threshold))
            else:
                reward = -2.0

            batch_rewards.append(reward)

            # Forward pass with gradients
            torch.set_grad_enabled(True)
            full_ids = outputs.sequences.clone().detach()
            gen_ids = generated_ids.clone().detach()

            # Policy forward
            policy_logits = policy(input_ids=full_ids).logits

            # Reference forward (no grad)
            with torch.no_grad():
                ref_logits = ref_model(input_ids=full_ids).logits

            # Get log probs of generated tokens
            gen_policy_logits = policy_logits[0, inputs.input_ids.shape[1]-1:-1, :]
            gen_ref_logits = ref_logits[0, inputs.input_ids.shape[1]-1:-1, :]

            # Compute KL divergence
            kl = compute_kl_divergence(gen_policy_logits, gen_ref_logits)
            batch_kls.append(kl.item())

            # Log probs for REINFORCE
            gen_probs = F.log_softmax(gen_policy_logits, dim=-1)
            token_log_probs = gen_probs.gather(1, gen_ids.unsqueeze(1)).squeeze(1)

            # Compute coherence penalty (perplexity under reference model)
            # High perplexity = gibberish = bad
            ref_ppl = compute_perplexity(
                gen_ref_logits.unsqueeze(0),
                gen_ids.unsqueeze(0)
            )
            # Normalize: log(ppl) to make it more manageable, subtract baseline
            coherence_penalty = torch.log(ref_ppl) - 2.0  # baseline ~7 ppl = log(7) ≈ 2
            coherence_penalty = torch.clamp(coherence_penalty, min=0.0)  # only penalize high ppl

            # REINFORCE loss with KL penalty + coherence penalty
            # Loss = -reward * log_prob + kl_coef * KL + coherence_coef * coherence
            policy_loss = -reward * token_log_probs.mean()
            kl_loss = args.kl_coef * kl
            coh_loss = args.coherence_coef * coherence_penalty
            loss = policy_loss + kl_loss + coh_loss

            batch_loss = batch_loss + loss

        # Update
        optimizer.zero_grad()
        (batch_loss / args.batch_size).backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        mean_reward = sum(batch_rewards) / len(batch_rewards)
        mean_kl = sum(batch_kls) / len(batch_kls)
        total_rewards.append(mean_reward)
        total_kls.append(mean_kl)

        if (step + 1) % 10 == 0:
            avg_reward = sum(total_rewards[-10:]) / min(10, len(total_rewards))
            avg_kl = sum(total_kls[-10:]) / min(10, len(total_kls))
            if args.coherence_coef > 0:
                print(f"Step {step + 1}/{args.max_steps} | Reward: {mean_reward:+.4f} | Avg: {avg_reward:+.4f} | KL: {avg_kl:.4f} | Coh: on")
            else:
                print(f"Step {step + 1}/{args.max_steps} | Reward: {mean_reward:+.4f} | Avg: {avg_reward:+.4f} | KL: {avg_kl:.4f}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final avg reward: {sum(total_rewards[-10:]) / 10:+.4f}")
    print(f"Final avg KL: {sum(total_kls[-10:]) / 10:.4f}")
    print("=" * 60)

    # Save model
    output_dir = Path("outputs/binoculars-kl")
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel saved to: {output_dir}")


if __name__ == "__main__":
    main()
