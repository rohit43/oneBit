from mlx_lm import load

MODEL = "prism-ml/Bonsai-8B-mlx-1bit"

print(f"Downloading {MODEL}...")
model, tokenizer = load(MODEL)
print("Done.")