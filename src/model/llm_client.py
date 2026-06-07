"""大模型客户端 — 封装本地 LLM 推理接口。

支持的 provider：
- transformers: 直接加载 HuggingFace safetensors 模型（推荐，无需 Ollama）
- ollama: 通过 Ollama API 调用
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMClient:
    """本地 LLM 客户端，提供 generate 方法。"""

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        device: str = "auto",
    ):
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._load_model()

    def _load_model(self):
        print(f"正在加载模型: {self.model_path} (device={self.device}) ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()
        print(f"模型加载完成，显存占用: {torch.cuda.max_memory_allocated()/1024**3:.1f}GB" if self.device == "cuda" else "模型加载完成 (CPU)")

    def generate(self, prompt: str) -> str:
        """调用本地 LLM 生成文本。"""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature if self.temperature > 0 else 1.0,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        result = self.tokenizer.decode(generated, skip_special_tokens=True)
        return result.strip()
