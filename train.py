# train_with_trainer.py
import torch
import json
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    PretrainedConfig
)
from datasets import Dataset, load_dataset
from DecoderOnlyModel import DecoderOnlyModelDecoder, DecoderOnlyModelConfig  # 你的实现
from tokenization_qwen2_fast import Qwen2TokenizerFast

# ========== 1. 加载配置和模型（从头初始化） ==========
config = DecoderOnlyModelConfig.from_json_file('./config.json')
# 或者：config = DecoderOnlyModelConfig(vocab_size=151936, hidden_size=896, num_hidden_layers=24, ...)

model = DecoderOnlyModelDecoder(config)  # 随机初始化，不加载 safetensors
tokenizer = Qwen2TokenizerFast.from_pretrained('.')

print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# ========== 2. 准备数据 ==========


# def load_data(data_path, max_samples=100000):
#     """加载对话数据并渲染成文本"""
#     texts = []
#     with open(data_path, 'r', encoding='utf-8') as f:
#         for i, line in enumerate(f):
#             if i >= max_samples:
#                 break
#             data = json.loads(line)
#             conversations = data.get('conversations', data.get('messages', []))

#             # 使用 chat_template 渲染
#             text = tokenizer.apply_chat_template(
#                 conversations,
#                 tokenize=False,
#                 add_generation_prompt=False
#             )
#             texts.append(text)
#     return texts


# # 加载数据
# raw_texts = load_data('sft_data_10k.jsonl', max_samples=100000)
# Dataset.from_dict({"text": raw_texts})
dataset = load_dataset(
    "json", data_files='sft_data_10k.jsonl', split="train", streaming=True
)
# 创建 Dataset 对象（Hugging Face datasets 格式）
# dataset = None; 

# Tokenize（不创建 labels，DataCollator 会处理）


def tokenize_function(examples):
    # texts = examples
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,
        padding=False,  # 动态 padding，由 data_collator 处理
        return_special_tokens_mask=False,
    )
    return outputs


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)

# ========== 3. DataCollator（关键） ==========
# DataCollatorForLanguageModeling 会自动：
# 1. 将 input_ids 复制为 labels
# 2. 将 labels 向右 shift 一位（预测下一个 token）
# 3. 处理 padding 并设置 -100
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # 因果语言模型（不是 BERT 的 Masked LM）
    pad_to_multiple_of=8,  # 方便 fp16 加速
)

# ========== 4. 训练参数 ==========
args = TrainingArguments(
    max_steps = int(10e4),
    output_dir="./qwen2_pretrain",
    per_device_train_batch_size=4,      # 根据显存调整
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,      # 有效 batch = 4*8 = 32
    num_train_epochs=3,                 # 10万条数据可以跑 3-5 轮
    learning_rate=5e-4,                 # 预训练可以用大点学习率
    weight_decay=0.1,
    warmup_ratio=0.03,                  # 3% warmup
    lr_scheduler_type="cosine",
    logging_steps=100,
    save_steps=1000,
    save_total_limit=3,                 # 最多保留 3 个 checkpoint
    fp16=True,                          # 混合精度
    tf32=True,                          # Ampere 以上显卡加速
    dataloader_num_workers=4,
    remove_unused_columns=False,        # 必须设为 False，否则 HF 会删掉我们的数据列
    gradient_checkpointing=True,        # 省显存（会慢一点）
    optim="adamw_torch_fused",  # fused AdamW（更快）
    report_to=["tensorboard"],          # 或 "wandb"
)

# ========== 5. 创建 Trainer ==========
trainer = Trainer(
    model=model,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
    # eval_dataset=...  # 如果有验证集可以加上
)

# ========== 6. 开始训练 ==========
trainer.train()

# 保存最终模型
trainer.save_model("./qwen2_pretrain_final")
tokenizer.save_pretrained("./qwen2_pretrain_final")
