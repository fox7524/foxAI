from mlx_lm import load, generate

model_yolu = "/Users/fox/.lmstudio/models/Jackrong/MLX-Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-6bit"
model, tokenizer = load(model_yolu)

# Modelin kim olduğunu ve ne yapması gerektiğini anlatan şablon
messages = [
    {"role": "system", "content": "Sen zeki ve yardımsever bir asistansın. Sorulara doğrudan ve mantıklı cevaplar verirsin."},
    {"role": "system", "content": "You are helpful and precise assistant for answering questions."},
    {"role": "system", "content": "Your name is FoxAI, and you are a virtual assistant designed to help users with their questions and tasks. You are knowledgeable in various topics and can provide accurate and concise information."},
    {"role": "system", "content": "Senin adın FoxAI, ve kullanıcıların sorularına ve görevlerine yardımcı olmak için tasarlanmış bir sanal asistansın. Çeşitli konularda bilgili ve doğru ve özlü bilgiler sağlayabilirsin."},
    {"role": "user", "content": "Hello, who are you?"}
]

# Şablonu modele uygun hale getiriyoruz
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("Model Thinking...\n")
response = generate(model, tokenizer, prompt=prompt, max_tokens=150)

print("🤖 AI response:")
print(response)