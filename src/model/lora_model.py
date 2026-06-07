"""LoRA 模型封装 — 加载基座模型+LoRA适配器进行推理

支持两种模式:
- base: 纯基座模型推理（用于对比）
- lora: 基座+LoRA适配器推理
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


class LoraModelClient:
    """LoRA增强的大模型客户端，提供 generate 方法。"""

    def __init__(
        self,
        model_path: str = "D:/llm_models/qwen/Qwen2-7B-Instruct",
        lora_path: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        load_in_4bit: bool = True,
    ):
        self.model_path = model_path
        self.lora_path = lora_path
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._load_model(load_in_4bit)

    def load_lora(self, lora_path: str):
        """在已加载的基座模型上加载并合并 LoRA 适配器。"""
        print(f"加载 LoRA 适配器: {lora_path}")
        self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model = self.model.merge_and_unload()
        self.model.eval()
        print("LoRA 适配器已合并")

    def _load_model(self, load_in_4bit):
        device_map = "auto" if torch.cuda.is_available() else "cpu"

        load_kwargs = {
            "device_map": device_map,
            "trust_remote_code": True,
        }

        if load_in_4bit and torch.cuda.is_available():
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            print(f"以4-bit量化加载基座模型: {self.model_path}")
        else:
            load_kwargs["torch_dtype"] = torch.float32
            print(f"加载基座模型(全精度): {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, **load_kwargs
        )

        # 加载 LoRA 适配器（如果提供）
        if self.lora_path:
            self.load_lora(self.lora_path)

        self.model.eval()

        gpu_mem = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        print(f"模型加载完成 (GPU显存: {gpu_mem:.1f}GB)" if gpu_mem > 0 else "模型加载完成 (CPU)")

    def generate(self, prompt: str) -> str:
        """调用模型生成文本。"""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature if self.temperature > 0 else 1.0,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
