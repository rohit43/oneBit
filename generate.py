from mlx_lm import load, generate

model, tokenizer = load("prism-ml/Bonsai-8B-mlx-1bit")

prompt = "What is Causal Inference?"
print(f"Prompt: {prompt}\n")

response = generate(model, tokenizer, prompt=prompt, max_tokens=256)
print(response)