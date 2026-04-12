from mlx_lm import load, generate

model_yolu = "/Users/fox/.lmstudio/models/Jackrong/MLX-Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-6bit"
model, tokenizer = load(model_yolu)

# Modelin kim olduğunu ve ne yapması gerektiğini anlatan şablon
messages = [
    {"role": "system", "content": "Sen zeki ve yardımsever bir asistansın. Sorulara doğrudan ve mantıklı cevaplar verirsin."},
    {"role": "user", "content": "Hello, how are you?"}
]

# Şablonu modele uygun hale getiriyoruz
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("Model konuşuyor...\n")
response = generate(model, tokenizer, prompt=prompt, max_tokens=150)

print("🤖 AI Cevabı:")
print(response)