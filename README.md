# Attention Knockout

An interactive mechanistic interpretability tool for GPT-2. Type a prompt, watch the next-token probability distribution, then click any of the 144 attention heads to zero out its contribution and observe how the output shifts in real time.

Built under Eunice Labs.

---

## What It Does

GPT-2 small has 12 layers and 12 attention heads per layer. Each head learns a different function during training -- some track syntactic dependencies, some resolve coreference, some are positional. Most people have read about this. This tool lets you see it directly.

For any prompt you type:

- The **head grid** shows all 144 heads colored by precomputed importance (KL divergence). Hotter color means the head has more influence on the output distribution.
- **Left-clicking** a head knocks it out by zeroing its 64-dimensional slice in the multi-head output during the forward pass. The probability chart updates immediately.
- **Right-clicking** a head opens its attention pattern -- the raw seq x seq softmax weight matrix showing what each token attends to.
- The **probability panel** shows base probabilities alongside the current (post-knockout) probabilities side by side. Bars that rise turn green, bars that fall turn red.

No model weights are modified. Knockouts are implemented via PyTorch forward hooks that are registered and removed on each forward pass.

---

## Setup

**Requirements**

- Python 3.9+
- ~500 MB disk space for GPT-2 weights (downloaded automatically on first run)
- No GPU required

**Install**

```bash
pip install -r requirements.txt
```

**Run**

```bash
uvicorn backend:app --reload --port 8000
```

Then serve the frontend:

```bash
python -m http.server 3000
```

Open `http://localhost:3000` in your browser.

The backend will print `Precomputing importance matrix...` on startup. This takes 30-60 seconds on CPU and runs once per server session.

---

## Project Structure

```
attention-knockout/
├── backend.py        FastAPI server, model logic, all three API endpoints
├── index.html        Single-file frontend, no build step, vanilla JS + Chart.js
├── requirements.txt  Python dependencies
└── README.md
```

---

## How the Knockout Works

The hook targets `model.transformer.h[layer].attn`, which outputs the concatenated multi-head result before the W_O projection. Each head occupies a contiguous 64-dimensional slice of the 768-dimensional output vector. Zeroing that slice is equivalent to removing the head's contribution to the residual stream entirely.

```python
def make_hook(head_idx):
    def hook(module, input, output):
        out = output[0].clone()
        start = head_idx * HEAD_DIM  # HEAD_DIM = 64
        out[:, :, start:start + HEAD_DIM] = 0.0
        return (out,) + output[1:]
    return hook
```

This is the same intervention used in Michel et al. (2019).

---

## Importance Score

At startup, the server computes a 12x12 importance matrix. For every head and every reference prompt, it measures the KL divergence between the base output distribution and the distribution after knocking out that head. The scores are averaged across prompts and normalized to [0, 1].

The five reference prompts were chosen because they exercise known head specializations: coreference resolution, subject-verb agreement, induction, long-range dependency, and factual recall.

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/probs` | Run a forward pass with specified heads knocked out. Returns top-k token probabilities. |
| GET | `/importance` | Return the precomputed 12x12 KL importance matrix. |
| GET | `/attention_pattern` | Return the seq x seq attention weight matrix for a specific head. |

**POST /probs**
```json
{
  "prompt": "The cat sat on the",
  "knocked_out_heads": [[3, 5], [7, 2]],
  "top_k": 20
}
```

**GET /attention_pattern**
```
?prompt=The cat sat on the&layer=4&head=6
```

---

## Prompts That Show Interesting Behavior

| Prompt | What to look for |
|--------|-----------------|
| `The trophy didn't fit in the suitcase because it was too` | Swap "trophy" and "suitcase" -- watch which heads change their resolution of "it" |
| `The keys to the cabinet are on the` | Heads that enforce plural subject agreement |
| `The cat sat on the mat. The dog sat on the` | Induction heads -- pattern [A][B]...[A] predicts [B] |
| `The lawyer said the doctor believes the patient was` | Long-range dependency heads in layers 8-11 |
| `In France, the capital city is` | Factual recall -- mostly MLP-driven but some heads contribute |

---

## Background

This tool implements the head ablation methodology from:

- **Are Sixteen Heads Really Better than One?** (Michel et al., 2019) -- direct precursor, attention head pruning via zeroing
- **A Mathematical Framework for Transformer Circuits** (Elhage et al., Anthropic 2021) -- theoretical grounding for residual stream interventions
- **In-context Learning and Induction Heads** (Olsson et al., 2022) -- why induction heads matter
- **Interpretability in the Wild** (Wang et al., 2022) -- full circuit reverse-engineering example

---

## Extension Ideas

- **Head atlas** -- systematic knockout of every head across 100 diverse prompts, producing a publishable map of GPT-2 head functions
- **Cross-model comparison** -- run the same analysis on GPT-2 medium and large, check if the same heads learn the same functions
- **Activation patching** -- instead of zeroing, patch a head's output from a different prompt to isolate what information it carries (causal tracing)
- **Circuit discovery** -- identify which combinations of heads together implement a specific behavior such as indirect object identification
