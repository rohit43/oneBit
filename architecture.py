from mlx_lm import load

model, tokenizer = load("prism-ml/Bonsai-8B-mlx-1bit")
print(model)