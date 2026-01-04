#!/usr/bin/env python3
"""
Quick evaluation of trained model against Binoculars.

Usage:
    uv run scripts/eval.py
    uv run scripts/eval.py --model outputs/binoculars-rl
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from binoculars import Binoculars
from binoculars.detector import BINOCULARS_ACCURACY_THRESHOLD


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="outputs/binoculars-rl", help="Model path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("Loading Binoculars detector...")
    bino = Binoculars(
        observer_name_or_path="Qwen/Qwen2.5-0.5B",
        performer_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        use_bfloat16=True,
    )
    threshold = BINOCULARS_ACCURACY_THRESHOLD
    print(f"Threshold: {threshold:.4f}\n")

    prompts = [
        "Once upon a time",
        "The scientist discovered",
        "In a small village",
        "My grandmother always said",
        "The robot opened its eyes",
    ]

    print("=" * 70)
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        score = bino.compute_score(text)
        label = "HUMAN" if score >= threshold else "AI"

        print(f"Score: {score:.3f} [{label}]")
        print(f"Text: {text[:150]}{'...' if len(text) > 150 else ''}")
        print("-" * 70)


if __name__ == "__main__":
    main()
