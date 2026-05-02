import json
import os
import re
import subprocess
import sys
import time
import shutil
from typing import List

def _presplit_text(text: str, max_seq_length: int, batch_size: int) -> list[str]:
    t = text or ""
    max_seq_length = int(max_seq_length) if int(max_seq_length) > 0 else 512
    batch_size = int(batch_size) if int(batch_size) > 0 else 1
    limit_chars = int(max_seq_length * 4)
    eff_limit = max(512, int(limit_chars / max(1, batch_size)))
    if len(t) <= eff_limit:
        return [t]
    if "<|im_start|>" in t and "<|im_end|>" in t:
        blocks = re.findall(r"<\|im_start\|>[\s\S]*?<\|im_end\|>\n?", t)
        if not blocks:
            blocks = [t]
        system_prefix = ""
        rest = blocks
        if blocks and blocks[0].startswith("<|im_start|>system"):
            system_prefix = blocks[0]
            rest = blocks[1:]
        out: list[str] = []
        i = 0
        while i < len(rest):
            cur = system_prefix
            j = i
            while j < len(rest) and len(cur) + len(rest[j]) <= eff_limit:
                cur += rest[j]
                j += 1
            if cur == system_prefix:
                b = rest[i]
                k = 0
                while k < len(b):
                    seg = b[k : k + eff_limit]
                    out.append(system_prefix + seg)
                    k += eff_limit
                i += 1
                continue
            out.append(cur)
            i = j
        return [s for s in out if s.strip()]
    parts = re.split(r"\n\s*\n", t)
    acc = ""
    out: list[str] = []
    for p in parts:
        p = (p or "").strip()
        if not p:
            continue
        cand = (acc + ("\n\n" if acc else "") + p) if acc else p
        if len(cand) <= eff_limit:
            acc = cand
            continue
        if acc:
            out.append(acc)
            acc = ""
        if len(p) <= eff_limit:
            acc = p
            continue
        k = 0
        while k < len(p):
            out.append(p[k : k + eff_limit])
            k += eff_limit
    if acc:
        out.append(acc)
    return [s for s in out if s.strip()]

def _presplit_jsonl_file(fp: str, max_seq_length: int, batch_size: int) -> int:
    if not fp or not os.path.isfile(fp):
        return 0
    changed = 0
    out_lines: list[str] = []
    with open(fp, "r", encoding="utf-8") as f:
        for ln in f.read().splitlines():
            s = (ln or "").strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                obj = {"text": s}
            if not isinstance(obj, dict) or "text" not in obj:
                out_lines.append(json.dumps(obj, ensure_ascii=False))
                continue
            text = str(obj.get("text") or "")
            pieces = _presplit_text(text, max_seq_length=max_seq_length, batch_size=batch_size)
            if len(pieces) > 1:
                changed += 1
            for p in pieces:
                obj2 = dict(obj)
                obj2["text"] = p
                out_lines.append(json.dumps(obj2, ensure_ascii=False))
    tmp = fp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for ln in out_lines:
            if ln.strip():
                f.write(ln + "\n")
    os.replace(tmp, fp)
    return changed

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

    def start_training(
        self,
        batch_size=2,
        num_layers=16,
        iters=100,
        dataset_path=None,
        adapter_path=None,
        config_path=None,
        resume_adapter_file: str | None = None,
    ) -> subprocess.Popen:
        """Starts the MLX LoRA training loop as a non-blocking subprocess."""
        data_dir = dataset_path if dataset_path else self.dataset_dir
        try:
            if os.environ.get("LOKUMAI_FT_PRESPLIT", "1") != "0":
                max_seq = int(os.environ.get("LOKUMAI_FT_MAX_SEQ_LENGTH", "512").strip() or "512")
                _presplit_jsonl_file(os.path.join(os.path.abspath(data_dir), "train.jsonl"), max_seq, int(batch_size))
                _presplit_jsonl_file(os.path.join(os.path.abspath(data_dir), "valid.jsonl"), max_seq, int(batch_size))
        except Exception:
            pass
        cmd = [sys.executable, "-m", "mlx_lm", "lora", "--model", self.model_path, "--train", "--data", data_dir]
        cmd += ["--batch-size", str(batch_size), "--num-layers", str(num_layers), "--iters", str(iters)]
        if resume_adapter_file:
            cmd += ["--resume-adapter-file", str(resume_adapter_file)]
        if os.environ.get("LOKUMAI_FT_GRAD_CHECKPOINT", "1") != "0":
            cmd += ["--grad-checkpoint"]
        val_batches = os.environ.get("LOKUMAI_FT_VAL_BATCHES", "1").strip()
        if val_batches:
            cmd += ["--val-batches", str(val_batches)]
        steps_per_eval = os.environ.get("LOKUMAI_FT_STEPS_PER_EVAL", "200").strip()
        if steps_per_eval:
            cmd += ["--steps-per-eval", str(steps_per_eval)]
        max_seq = os.environ.get("LOKUMAI_FT_MAX_SEQ_LENGTH", "512").strip()
        if max_seq:
            cmd += ["--max-seq-length", str(max_seq)]
        clear_thr = os.environ.get("LOKUMAI_FT_CLEAR_CACHE_THRESHOLD", "2.0").strip()
        if clear_thr:
            cmd += ["--clear-cache-threshold", str(clear_thr)]
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

    def start_validation(self, dataset_path: str, adapter_path: str, config_path: str | None = None) -> subprocess.Popen:
        """
        Runs a post-training evaluation pass using the dataset's valid.jsonl as test.jsonl.
        This allows "train first, validate later" without running validation during training.
        """
        data_dir = dataset_path if dataset_path else self.dataset_dir
        valid_fp = os.path.join(os.path.abspath(data_dir), "valid.jsonl")
        if not os.path.isfile(valid_fp):
            raise RuntimeError("valid.jsonl not found in dataset directory.")

        ts = time.strftime("%Y%m%d_%H%M%S")
        eval_dir = os.path.abspath(os.path.join("lora_data", "validate_only", f"run_{ts}"))
        os.makedirs(eval_dir, exist_ok=True)
        shutil.copyfile(valid_fp, os.path.join(eval_dir, "test.jsonl"))
        try:
            if os.environ.get("LOKUMAI_FT_PRESPLIT", "1") != "0":
                max_seq = int(os.environ.get("LOKUMAI_FT_MAX_SEQ_LENGTH", "512").strip() or "512")
                _presplit_jsonl_file(os.path.join(eval_dir, "test.jsonl"), max_seq, 1)
        except Exception:
            pass

        cmd = [sys.executable, "-m", "mlx_lm", "lora", "--model", self.model_path, "--data", eval_dir, "--test"]
        test_batches = os.environ.get("LOKUMAI_FT_TEST_BATCHES", "1").strip()
        if test_batches:
            cmd += ["--test-batches", str(test_batches)]
        max_seq = os.environ.get("LOKUMAI_FT_MAX_SEQ_LENGTH", "512").strip()
        if max_seq:
            cmd += ["--max-seq-length", str(max_seq)]
        clear_thr = os.environ.get("LOKUMAI_FT_CLEAR_CACHE_THRESHOLD", "2.0").strip()
        if clear_thr:
            cmd += ["--clear-cache-threshold", str(clear_thr)]
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
