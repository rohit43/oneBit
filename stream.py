import time
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

model, tokenizer = load("prism-ml/Bonsai-8B-mlx-1bit")

prompt = "What is Causal Inference?"
print(f"Prompt: {prompt}\n")

sampler = make_sampler(temp=0.5, top_p=0.9)
token_count = 0
start = time.time()

for response in stream_generate(model, tokenizer, prompt=prompt, max_tokens=512, sampler=sampler):
    print(response.text, end="", flush=True)
    token_count += 1

elapsed = time.time() - start
print(f"\n\n{token_count} tokens in {elapsed:.1f}s — {token_count / elapsed:.1f} tok/s")