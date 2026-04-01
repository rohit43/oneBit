import mlx.utils
from mlx_lm import load

model, tokenizer = load("prism-ml/Bonsai-8B-mlx-1bit")

total_params = sum(x.size for _, x in mlx.utils.tree_flatten(model.parameters()))
num_layers = len(model.model.layers)

layer = model.model.layers[0]
q_proj = layer.self_attn.q_proj
k_proj = layer.self_attn.k_proj
gate_proj = layer.mlp.gate_proj

print(f"Model:              Bonsai-8B-mlx-1bit")
print(f"Base architecture:  Qwen3-8B")
print(f"Parameters (1-bit): {total_params / 1e9:.2f}B")
print(f"Parameters (orig):  8.19B")
print(f"Quantization:       {q_proj.bits}-bit, group_size={q_proj.group_size}")
print(f"Bits per weight:    1.25 bpw")
print(f"Layers:             {num_layers}")
print(f"Hidden size:        {q_proj.weight.shape[1] * 32}")
print(f"Q heads:            {q_proj.weight.shape[1] * 32 // 128}")
print(f"KV heads:           {k_proj.weight.shape[1] * 32 // 128}")
print(f"MLP intermediate:   {gate_proj.weight.shape[1] * 32}")
print(f"Vocab size:         {model.model.embed_tokens.weight.shape[1] * 32}")
print(f"Context length:     65,536")
print(f"Tokenizer:          {type(tokenizer).__name__}")