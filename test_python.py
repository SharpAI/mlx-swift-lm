from mlx_lm import load, generate

model, tokenizer = load("mlx-community/gemma-4-e4b-it-4bit")
response = generate(model, tokenizer, prompt="The answer to 2+2 is", verbose=True, max_tokens=20)
print("RESPONSE:", response)
