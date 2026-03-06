import requests
import json

# Ensure Ollama is running in the background before executing this
OLLAMA_URL = "http://localhost:11434/api/generate"

def ask_ollama(model, prompt):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            print(f"Error from {model}:", response.text)
            return ""
    except Exception as e:
        print(f"Connection failed: {e}")
        return ""

def generate_perfect_data(instruction):
    print(f"\n[1] DeepSeek is writing the code for: {instruction}...")
    
    # DeepSeek acts as the Lead Coder
    coder_prompt = f"You are an expert software engineer. Write a highly optimized, bug-free script for the following request. ONLY output the code, no explanations.\nRequest: {instruction}"
    draft_code = ask_ollama("deepseek-coder:6.7b", coder_prompt)
    
    print("[2] CodeLlama is reviewing the code...")
    
    # CodeLlama acts as the Senior Reviewer
    reviewer_prompt = f"You are a strict senior code reviewer. Review the following code for bugs, logic errors, or memory leaks. If it is perfect, output exactly 'PASS'. If there are errors, provide the corrected code.\n\nCode to review:\n{draft_code}"
    review_feedback = ask_ollama("codellama:7b", reviewer_prompt)
    
    final_code = draft_code
    if "PASS" not in review_feedback.upper():
        print("[!] CodeLlama found issues and revised the code.")
        final_code = review_feedback
    else:
        print("[+] CodeLlama approved the code.")

    # Save to your training vault in JSONL format
    dataset_entry = {
        "instruction": instruction,
        "output": final_code.strip()
    }
    
    with open("training_vault.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dataset_entry) + "\n")
        
    print("[3] Successfully saved to database\n")

# Start building your dataset today
if __name__ == "__main__":
    print("=== foxAI: Dual-Brain Training Generator ===")
    while True:
        user_req = input("Enter a coding task to generate training data (or 'exit'): ")
        if user_req.lower() == 'exit':
            break
        generate_perfect_data(user_req)