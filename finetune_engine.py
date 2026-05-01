import json
import os
import subprocess
import sys
from typing import List

class FinetuneEngine:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.dataset_dir = "lora_data"
        os.makedirs(self.dataset_dir, exist_ok=True)
            
    def prepare_dataset(self, text_chunks: List[str]):
        """Converts raw text chunks into a train/valid JSONL dataset for MLX lora."""
        train_path = os.path.join(self.dataset_dir, "train.jsonl")
        valid_path = os.path.join(self.dataset_dir, "valid.jsonl")
        
        split_idx = int(len(text_chunks) * 0.9)
        train_chunks = text_chunks[:split_idx]
        valid_chunks = text_chunks[split_idx:]
        
        if not train_chunks:
            train_chunks = text_chunks
            valid_chunks = text_chunks[:1]
            
        with open(train_path, "w", encoding="utf-8") as ft, open(valid_path, "w", encoding="utf-8") as fv:
            for chunk in train_chunks:
                ft.write(json.dumps({"text": f"Instruction: Analyze the following knowledge.\n\nKnowledge: {chunk}\n\nResponse: Understood."}) + "\n")
            for chunk in valid_chunks:
                fv.write(json.dumps({"text": f"Instruction: Analyze the following knowledge.\n\nKnowledge: {chunk}\n\nResponse: Understood."}) + "\n")
                
        return train_path, valid_path

    def build_ask_before_acting_dataset(self, qa_pairs: List[dict]):
        """
        Specialized builder for the 50 hand-written 'Ask Before Acting' pairs.
        Each dictionary should have 'user' and 'assistant' keys.
        """
        train_path = os.path.join(self.dataset_dir, "ask_before_acting_train.jsonl")
        
        with open(train_path, "w", encoding="utf-8") as ft:
            for pair in qa_pairs:
                # ChatML formatting for model explicit behavior
                formatted_text = f"<|im_start|>user\n{pair['user']}<|im_end|>\n<|im_start|>assistant\n{pair['assistant']}<|im_end|>\n"
                ft.write(json.dumps({"text": formatted_text}) + "\n")
                
        return train_path

    def start_training(self, batch_size=2, num_layers=16, iters=100, dataset_path=None, adapter_path=None, config_path=None) -> subprocess.Popen:
        """Starts the MLX LoRA training loop as a non-blocking subprocess."""
        data_dir = dataset_path if dataset_path else self.dataset_dir
        cmd = [sys.executable, "-m", "mlx_lm", "lora", "--model", self.model_path, "--train", "--data", data_dir]
        cmd += ["--batch-size", str(batch_size), "--num-layers", str(num_layers), "--iters", str(iters)]
        if os.environ.get("LOKUMAI_FT_GRAD_CHECKPOINT", "1") != "0":
            cmd += ["--grad-checkpoint"]
        if adapter_path:
            cmd += ["--adapter-path", str(adapter_path)]
        if config_path:
            cmd += ["--config", str(config_path)]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=(sys.platform != "win32"),
        )
        return process
