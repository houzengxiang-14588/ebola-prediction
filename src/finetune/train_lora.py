"""QLoRA 微调训练器 — 对 Qwen2-7B-Instruct 进行 4-bit LoRA 微调

显存预算 (RTX 3070 Ti 8GB):
- 4-bit base model: ~4 GB
- LoRA params (r=16): ~20 MB
- Gradients + optimizer: ~80 MB
- Activations (seq=512, gc on): ~1.5 GB
- Total: ~6 GB (在8GB内可行)
"""

import os, sys, json, io
from pathlib import Path

# 修复 TRL 在中文 Windows 上的 GBK 编码问题
import locale
locale.getpreferredencoding = lambda: 'UTF-8'
import pathlib
_orig_read_text = pathlib.Path.read_text
def _read_text_utf8(self, encoding='utf-8', errors='ignore', **kwargs):
    return _orig_read_text(self, encoding=encoding, errors=errors, **kwargs)
pathlib.Path.read_text = _read_text_utf8

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling


def load_model_4bit(model_path: str, device_map="auto"):
    """以4-bit量化加载基座模型。"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",  # PyTorch原生，无需flash-attn
    )
    model = prepare_model_for_kbit_training(model)
    return model


def apply_lora(model, r=16, alpha=32, dropout=0.05):
    """对模型应用LoRA适配器。"""
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_training_data(data_path="data/training/samples.jsonl", tokenizer=None):
    """加载训练数据并转换为HuggingFace Dataset格式。

    返回 tokenized dataset，可直接用于 Trainer。
    """
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # 格式化为 chat template 格式
    texts = []
    for r in records:
        text = (
            f"<|im_start|>system\n你是埃博拉病毒疫情预测助手，基于历史数据预测未来趋势。"
            f"以准确JSON格式输出预测结果。<|im_end|>\n"
            f"<|im_start|>user\n{r['instruction']}<|im_end|>\n"
            f"<|im_start|>assistant\n{r['output']}<|im_end|>"
        )
        texts.append(text)

    if tokenizer is None:
        return Dataset.from_list([{"text": t} for t in texts])

    # Tokenize
    def tokenize_fn(text):
        return tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding=False,
            return_tensors=None,
        )

    dataset = Dataset.from_list([{"text": t} for t in texts])
    dataset = dataset.map(
        lambda x: tokenize_fn(x["text"]),
        remove_columns=["text"],
        desc="Tokenizing",
    )
    return dataset


def train(
    model_path: str = "D:/llm_models/qwen/Qwen2-7B-Instruct",
    data_path: str = "data/training/samples.jsonl",
    output_dir: str = "lora_weights",
    batch_size: int = 1,
    gradient_accumulation: int = 4,
    learning_rate: float = 2e-4,
    epochs: int = 3,
    max_seq_length: int = 512,
    save_steps: int = 50,
):
    """执行QLoRA微调训练。

    Args:
        model_path: 基座模型路径
        data_path: 训练数据JSONL路径
        output_dir: LoRA权重输出目录
    """
    print(f"加载基座模型: {model_path}")
    model = load_model_4bit(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("应用 LoRA 适配器...")
    model = apply_lora(model)

    print(f"加载训练数据: {data_path}")
    dataset = load_training_data(data_path, tokenizer=tokenizer)
    print(f"训练样本数: {len(dataset)}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        logging_steps=10,
        save_steps=save_steps,
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_8bit",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    # 使用原生 HuggingFace Trainer (避免 TRL 的编码问题)
    from transformers import Trainer
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("开始训练...")
    trainer.train()

    # 保存最终模型
    final_path = os.path.join(output_dir, "final")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"LoRA 权重已保存到: {final_path}")
    return trainer


if __name__ == "__main__":
    os.chdir("D:/Ebola")
    train()
