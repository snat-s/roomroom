import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from binoculars import Binoculars
from binoculars.detector import BINOCULARS_ACCURACY_THRESHOLD


def main():
    print("Loading models...")

    # Policy model (the one we'd train)
    policy_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(policy_name)
    policy = AutoModelForCausalLM.from_pretrained(
        policy_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Binoculars detector
    bino = Binoculars(
        observer_name_or_path="Qwen/Qwen2.5-0.5B",
        performer_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        use_bfloat16=True,
    )

    threshold = BINOCULARS_ACCURACY_THRESHOLD
    print(f"Threshold: {threshold:.4f}")

    # Test prompts
    prompts = [
        "Once upon a time, there was a little rabbit who",
        "The scientist looked at the data and realized that",
        "In a small village by the sea, an old fisherman",
        "The robot opened its eyes for the first time and",
        "My grandmother always told me that the secret to happiness",
    ]

    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    for i, prompt in enumerate(prompts):
        # Generate completion
        messages = [{"role": "user", "content": f"Continue this story in 2-3 sentences: {prompt}"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(policy.device)

        with torch.no_grad():
            outputs = policy.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        completion = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Score with Binoculars
        score = bino.compute_score(completion)
        reward = max(-2.0, min(2.0, score - threshold))
        prediction = "HUMAN" if score >= threshold else "AI"

        # Display
        print(f"\n[{i+1}] Prompt: \"{prompt}...\"")
        print(f"    Completion: \"{completion[:150]}{'...' if len(completion) > 150 else ''}\"")
        print(f"    Score: {score:.4f} | Threshold: {threshold:.4f} | Reward: {reward:+.4f} | -> {prediction}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
