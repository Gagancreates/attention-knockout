from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from torch.nn.functional import kl_div

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model setup — loaded once at startup, stays in memory
# ---------------------------------------------------------------------------
print("Loading GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
model.eval()

N_LAYERS = model.config.n_layer    # 12
N_HEADS  = model.config.n_head     # 12
HEAD_DIM = model.config.n_embd // N_HEADS  # 64

REFERENCE_PROMPTS = [
    "The trophy didn't fit in the suitcase because it was too",
    "The keys to the cabinet are on the",
    "The cat sat on the",
    "John said that Mary believes the answer is",
    "The company announced that their new product will be",
]


# ---------------------------------------------------------------------------
# Core: run a forward pass with selected heads zeroed out
# ---------------------------------------------------------------------------
def get_probs(prompt: str, knocked_out_heads: list[tuple[int, int]], top_k: int = 20):
    inputs = tokenizer(prompt, return_tensors="pt")
    hooks = []

    def make_hook(head_idx: int):
        def hook(module, input, output):
            # output[0]: (batch, seq_len, n_embd) — concatenated multi-head output
            # before W_O projection
            out = output[0].clone()
            start = head_idx * HEAD_DIM
            out[:, :, start:start + HEAD_DIM] = 0.0
            return (out,) + output[1:]
        return hook

    for layer_idx, head_idx in knocked_out_heads:
        attn_module = model.transformer.h[layer_idx].attn
        h = attn_module.register_forward_hook(make_hook(head_idx))
        hooks.append(h)

    with torch.no_grad():
        outputs = model(**inputs)

    for h in hooks:
        h.remove()

    logits = outputs.logits[0, -1, :]
    probs  = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, top_k)

    return [
        {"token": tokenizer.decode(idx.item()), "prob": round(prob.item(), 4)}
        for idx, prob in zip(top_indices, top_probs)
    ]


# ---------------------------------------------------------------------------
# Core: capture the raw attention weight matrix for one head
# ---------------------------------------------------------------------------
def get_attention_pattern(prompt: str, layer_idx: int, head_idx: int):
    inputs = tokenizer(prompt, return_tensors="pt")
    captured = {}

    def hook(module, input, output):
        # output[1] is attention weights when output_attentions=True
        # shape: (batch, n_heads, seq_len, seq_len)
        if len(output) > 1 and output[1] is not None:
            captured["weights"] = output[1].detach()

    handle = model.transformer.h[layer_idx].attn.register_forward_hook(hook)

    with torch.no_grad():
        model(**inputs, output_attentions=True)

    handle.remove()

    attn_weights = captured["weights"][0, head_idx].tolist()
    tokens = [tokenizer.decode([t]) for t in inputs["input_ids"][0]]

    return {"tokens": tokens, "matrix": attn_weights}


# ---------------------------------------------------------------------------
# Startup: precompute per-head KL importance across reference prompts
# ---------------------------------------------------------------------------
def compute_importance_matrix() -> list[list[float]]:
    importance = [[0.0] * N_HEADS for _ in range(N_LAYERS)]

    for prompt in REFERENCE_PROMPTS:
        base = get_probs(prompt, [])
        base_tensor = torch.tensor([x["prob"] for x in base])

        for layer in range(N_LAYERS):
            for head in range(N_HEADS):
                knocked = get_probs(prompt, [(layer, head)])
                knocked_tensor = torch.tensor([x["prob"] for x in knocked])
                kl = kl_div(
                    knocked_tensor.log().clamp(min=-100),
                    base_tensor,
                    reduction="sum",
                ).item()
                importance[layer][head] += kl

    flat = [importance[l][h] for l in range(N_LAYERS) for h in range(N_HEADS)]
    max_val = max(flat) or 1.0
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            importance[l][h] = round(importance[l][h] / max_val, 4)

    return importance


print("Precomputing importance matrix (this takes ~30-60 s on CPU)...")
IMPORTANCE_MATRIX = compute_importance_matrix()
print("Ready.")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class ProbsRequest(BaseModel):
    prompt: str
    knocked_out_heads: list[list[int]] = []
    top_k: int = 20


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/probs")
def probs(req: ProbsRequest):
    result = get_probs(
        req.prompt,
        [tuple(h) for h in req.knocked_out_heads],
        req.top_k,
    )
    return {"result": result}


@app.get("/importance")
def importance():
    return {"matrix": IMPORTANCE_MATRIX}


@app.get("/attention_pattern")
def attention_pattern(prompt: str, layer: int, head: int):
    return get_attention_pattern(prompt, layer, head)
