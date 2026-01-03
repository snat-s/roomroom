# Binoculars: Complete Guide

## Overview

Binoculars is a **zero-shot AI text detector** that requires NO training data. It detects AI-generated text by comparing how two related LLMs "see" the same text.

**Paper**: [Spotting LLMs with Binoculars: Zero-Shot Detection of Machine-Generated Text](https://arxiv.org/abs/2401.12070)

---

## How It Works

### The Core Formula

```
Binoculars Score = Perplexity / Cross-Entropy
```

Where:
- **Perplexity**: How "surprised" the **performer model** (Falcon-7B-Instruct) is by the text
- **Cross-Entropy**: How much the **observer model** (Falcon-7B) and performer model disagree about the text

### Why This Works

1. **AI-generated text** follows predictable patterns that both models recognize
2. Both models were trained on similar data (Common Crawl, Pile, etc.)
3. The performer (instruction-tuned) has slightly different distributions than the observer (base model)
4. When text is AI-generated, **both models agree** → low cross-entropy → **lower score**
5. When text is human-written, models disagree more → higher cross-entropy → **higher score**

### The Decision Logic

From `detector.py:100-102`:
- If score **< threshold** → `"Most likely AI-generated"`
- If score **>= threshold** → `"Most likely human-generated"`

### Thresholds

| Mode | Threshold | Optimized For |
|------|-----------|---------------|
| Low FPR (default) | 0.8536 | Only 0.01% false positives |
| Accuracy | 0.9015 | Balanced F1 score |

---

## Architecture

### Models Used (Default)

- **Observer**: `tiiuae/falcon-7b` (~7B parameters)
- **Performer**: `tiiuae/falcon-7b-instruct` (~7B parameters)

Both models must share the same tokenizer.

### Code Flow

1. Text is tokenized using the observer's tokenizer
2. Both models compute logits for the text
3. `perplexity()` calculates how surprised the performer is
4. `entropy()` calculates cross-entropy between observer and performer distributions
5. Score = perplexity / cross-entropy
6. Compare to threshold for final prediction

---

## Repository Structure

```
Binoculars/
├── binoculars/               # Core detection module
│   ├── __init__.py           # Exports Binoculars class
│   ├── detector.py           # Main Binoculars class implementation
│   ├── metrics.py            # perplexity() and entropy() functions
│   └── utils.py              # Tokenizer consistency check
├── demo/
│   └── demo.py               # Gradio web interface
├── experiments/              # Research evaluation scripts
│   ├── run.py                # Benchmark evaluation
│   ├── utils.py              # Plotting and result saving
│   └── jobs.sh               # Example experiment commands
├── app.py                    # Gradio launcher
├── main.py                   # Basic usage example
├── setup.py                  # Package installation
├── requirements.txt          # Dependencies
├── README.md                 # Official documentation
└── LICENSE.md                # BSD 3-Clause License
```

---

## Hardware Requirements

| Configuration | VRAM Needed | Storage |
|---------------|-------------|---------|
| 2x Falcon-7B (bfloat16) | ~14GB | ~30GB download |
| 2x Falcon-7B (float32) | ~28GB | ~30GB download |
| CPU-only | ~16GB RAM | Very slow |

**Important**: This is NOT a small classifier. It requires running two full 7B models at inference time.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/ahans30/Binoculars.git
cd Binoculars

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Dependencies

```
sentencepiece
transformers[torch] (v4.31.0)
datasets
numpy
gradio
gradio_client
scikit-learn
seaborn
pandas
```

---

## Usage

### Basic Python Usage

```python
from binoculars import Binoculars

# Initialize detector (downloads models on first run)
bino = Binoculars()

# Check a single text
text = "Your suspicious text here..."
score = bino.compute_score(text)      # Returns float (e.g., 0.756)
result = bino.predict(text)           # Returns "Most likely AI-generated" or "Most likely human-generated"

print(f"Score: {score}")
print(f"Result: {result}")
```

### Batch Processing

```python
from binoculars import Binoculars

bino = Binoculars()

texts = [
    "First text to check...",
    "Second text to check...",
    "Third text to check..."
]

scores = bino.compute_score(texts)    # List of floats
results = bino.predict(texts)         # List of predictions

for text, score, result in zip(texts, scores, results):
    print(f"{result} (score: {score:.4f}): {text[:50]}...")
```

### Web Interface

```bash
python app.py
```

Launches a Gradio UI in your browser.

### Configuration Options

```python
bino = Binoculars(
    observer_name_or_path="tiiuae/falcon-7b",        # Observer model
    performer_name_or_path="tiiuae/falcon-7b-instruct",  # Performer model
    use_bfloat16=True,                                # Reduced precision (saves memory)
    max_token_observed=512,                           # Max tokens to process
    mode="low-fpr"                                    # "low-fpr" or "accuracy"
)

# Switch modes at runtime
bino.change_mode("accuracy")
```

---

## API Reference

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `compute_score(text)` | `str` or `list[str]` | `float` or `list[float]` | Raw binoculars score (lower = more likely AI) |
| `predict(text)` | `str` or `list[str]` | `str` or `list[str]` | Human-readable prediction |
| `change_mode(mode)` | `"low-fpr"` or `"accuracy"` | None | Switch threshold mode |

---

## Interpreting Scores

| Score Range | Interpretation |
|-------------|----------------|
| < 0.8536 | Most likely AI-generated (low-fpr mode) |
| < 0.9015 | Most likely AI-generated (accuracy mode) |
| >= threshold | Most likely human-generated |

**Lower scores = more likely AI-generated**

---

## Limitations

1. **Language**: Better at detecting English text than other languages
2. **Length**: Works best with 200-300 words (64-1000 tokens)
3. **Minimum tokens**: Needs at least ~64 tokens to work reliably
4. **Memorized content**: May misclassify famous quotes, constitutions, etc. as AI
5. **Not a product**: Academic implementation, should not be used without human supervision

---

## Lightweight Alternatives

Since Binoculars requires two 7B models, here are options for limited hardware:

### Option 1: Use the Hosted Demo
Free HuggingFace Spaces demo (no GPU needed):
```
https://huggingface.co/spaces/tomg-group-umd/Binoculars
```

### Option 2: Try Smaller Models
```python
bino = Binoculars(
    observer_name_or_path="gpt2",           # 124M params
    performer_name_or_path="gpt2-medium",   # 355M params
)
```
**Warning**: Thresholds won't work - you'd need to recalibrate.

### Option 3: Different Methods
For small trained classifiers, consider:
- RoBERTa-based detectors (GPTZero-style)
- GLTR (statistical features + classifier)
- Custom classifier trained on AI vs human text

---

## Core Implementation Details

### Perplexity Calculation (`metrics.py:9-27`)

```python
def perplexity(encoding, logits, median=False, temperature=1.0):
    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

    ppl = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) *
           shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
    return ppl.to("cpu").float().numpy()
```

### Cross-Entropy Calculation (`metrics.py:30-57`)

```python
def entropy(p_logits, q_logits, encoding, pad_token_id, ...):
    p_proba = softmax_fn(p_scores).view(-1, vocab_size)
    q_scores = q_scores.view(-1, vocab_size)

    ce = ce_loss_fn(input=q_scores, target=p_proba)
    # Mask padding and aggregate
    agg_ce = (ce * padding_mask).sum(1) / padding_mask.sum(1)
    return agg_ce
```

### Score Computation (`detector.py:87-96`)

```python
def compute_score(self, input_text):
    encodings = self._tokenize(batch)
    observer_logits, performer_logits = self._get_logits(encodings)

    ppl = perplexity(encodings, performer_logits)
    x_ppl = entropy(observer_logits, performer_logits, encodings, pad_token_id)

    binoculars_scores = ppl / x_ppl
    return binoculars_scores
```

---

## Citation

```bibtex
@misc{hans2024spotting,
      title={Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text},
      author={Abhimanyu Hans and Avi Schwarzschild and Valeriia Cherepanova and Hamid Kazemi and Aniruddha Saha and Micah Goldblum and Jonas Geiping and Tom Goldstein},
      year={2024},
      eprint={2401.12070},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

---

## License

BSD 3-Clause License
