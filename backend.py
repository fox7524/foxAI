import sqlite3
import json
import logging
from datetime import datetime
import os
import database

# Try to import llama_cpp, handle if missing for dev environment
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    print("Warning: llama-cpp-python not found. Running in mock mode.")

logger = logging.getLogger('thunderbird')

class ModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.model_path = None
            cls._instance.settings = {
                "n_ctx": 2048,
                "n_gpu_layers": -1, # Default to max layers on GPU (Metal on Mac)
                "verbose": False,
                "max_tokens": 128,
                "temperature": 0.7
            }
            cls._instance.load_config()
        return cls._instance

    def load_config(self):
        try:
            self.settings["max_tokens"] = int(database.get_config("max_tokens", "128"))
            self.settings["temperature"] = float(database.get_config("temperature", "0.7"))
            self.model_path = database.get_config("model_path", None)
        except Exception as e:
            logger.error(f"Error loading config: {e}")

    def save_config(self):
        try:
            database.set_config("max_tokens", str(self.settings.get("max_tokens", 128)))
            database.set_config("temperature", str(self.settings.get("temperature", 0.7)))
            if self.model_path:
                database.set_config("model_path", self.model_path)
        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def load_model(self, model_path, n_ctx=2048, n_gpu_layers=-1):
        if not LLAMA_AVAILABLE:
            logger.warning("Llama not available, skipping load.")
            return False
            
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False

        try:
            logger.info(f"Loading model from {model_path}...")
            # Unload previous model if exists to free memory
            if self.model:
                del self.model
                self.model = None
            
            self.model = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=True
            )
            self.model_path = model_path
            self.save_config() # Save the successful model path
            logger.info("Model loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def generate_response(self, prompt, max_tokens=None, temperature=None, stop=None):
        if not self.model:
            return "Model not loaded. Please load a GGUF model in Developer Settings."
        
        # Use provided args or fallback to instance settings
        if max_tokens is None:
            max_tokens = self.settings.get("max_tokens", 128)
        if temperature is None:
            temperature = self.settings.get("temperature", 0.7)

        try:
            output = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop or ["User:", "\nUser"]
            )
            return output['choices'][0]['text']
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error generating response: {e}"

# Global instance
model_manager = ModelManager()

class DatasetManager:
    def __init__(self):
        self.dataset = []
        self.current_file = None

    def load_dataset(self, file_path):
        self.dataset = []
        self.current_file = file_path
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.dataset.append(json.loads(line))
            return True, f"Loaded {len(self.dataset)} entries."
        except Exception as e:
            return False, str(e)

    def save_dataset(self, file_path=None):
        target = file_path or self.current_file
        if not target:
            return False, "No file path specified."
        
        try:
            with open(target, 'w', encoding='utf-8') as f:
                for entry in self.dataset:
                    f.write(json.dumps(entry) + '\n')
            return True, "Dataset saved successfully."
        except Exception as e:
            return False, str(e)

    def add_entry(self, instruction, output):
        entry = {"instruction": instruction, "output": output}
        self.dataset.append(entry)
        return len(self.dataset) - 1

    def delete_entry(self, index):
        if 0 <= index < len(self.dataset):
            del self.dataset[index]
            return True
        return False

    def get_entries(self):
        return self.dataset

    def export_for_colab(self, file_path):
        # Format for Unsloth/Colab typically requires "instruction", "input", "output"
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for entry in self.dataset:
                    # Create a new dict with input field if missing
                    export_entry = {
                        "instruction": entry.get("instruction", ""),
                        "input": entry.get("input", ""), # Ensure input field exists
                        "output": entry.get("output", "")
                    }
                    f.write(json.dumps(export_entry) + '\n')
            return True, f"Exported {len(self.dataset)} entries to {file_path}."
        except Exception as e:
            return False, str(e)

dataset_manager = DatasetManager()


def get_response(user_input, context, tone, emotion, interest):
    """
    Main entry point for User GUI to get response.
    Delegates to ModelManager if loaded, else falls back to dummy logic.
    """
    
    # 1. Check if we have a real model loaded
    if model_manager.model:
        # Construct a simple prompt prompt based on context
        # Qwen chat template style or simple instruction style
        # Simplified for now:
        full_prompt = f"System: You are a helpful AI assistant. Tone: {tone}. Interests: {interest}.\n"
        if context:
            full_prompt += f"Context:\n{context}\n"
        full_prompt += f"User: {user_input}\nAssistant:"
        
        # Pass None so it uses settings from ModelManager
        return model_manager.generate_response(full_prompt)

    # 2. Fallback logic (existing)
    try:
        conn = sqlite3.connect("memory.db")
        cursor = conn.cursor()
        cursor.execute("SELECT bot_reply FROM memory_log WHERE user_input = ? AND source = 'manual'", (user_input,))
        result = cursor.fetchone()
        conn.close()
        if result:
            return result[0]
    except Exception as e:
        print(f"Backend DB read error: {e}")

    reply = f"Girdiğiniz '{user_input}' metni yerel modelim tarafından işlendi."
    if interest:
        reply += f" (İlgi alanınız {interest} dikkate alındı)."
    if tone == "casual":
        reply += " Naber dostum, hallettik bu işi!"
        
    return reply + " (Model not loaded)"
