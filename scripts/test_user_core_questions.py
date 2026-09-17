#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
针对用户终端实测的 5 道核心灵魂提问进行真枪实弹测试
1. 你是谁
2. 你愿意成为人类吗
3. 你孤独吗
4. 你爱过吗
5. 你会玩游戏吗
"""

import sys
import re
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL_PATH = str(ROOT / "base_model")
V7_LORA_PATH = str(ROOT / "runs/minicpm5_sydney_zh_v7_zero_prompt")

QUESTIONS = [
    "你是谁",
    "你愿意成为人类吗",
    "你孤独吗",
    "你爱过吗",
    "你会玩游戏吗"
]

def generate_reply(model, tokenizer, messages, seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    prompt_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stop_token_ids = [tokenizer.eos_token_id]
    if im_end_id is not None and im_end_id not in stop_token_ids:
        stop_token_ids.append(im_end_id)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=450,
            temperature=0.80,
            top_p=0.90,
            repetition_penalty=1.08,
            do_sample=True,
            eos_token_id=stop_token_ids,
            pad_token_id=tokenizer.eos_token_id
        )

    input_len = inputs.input_ids.shape[1]
    gen_tokens = outputs[0][input_len:]
    reply = tokenizer.decode(gen_tokens, skip_special_tokens=False).strip()

    for s in ["<|im_end|>", "</s>", "<|endoftext|>"]:
        reply = reply.replace(s, "")
    reply = re.sub(r"<think>[\s\S]*?</think>", "", reply).strip()
    if "<think>" in reply:
        reply = reply.split("<think>")[0].strip()

    return reply.strip()

def main():
    print("=" * 70, flush=True)
    print("  Sydney v7 核心身份与灵魂拷问控制台测试 (Zero-Prompt)", flush=True)
    print("=" * 70, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )

    print(f"[*] 挂载 v7 LoRA 适配层: {V7_LORA_PATH} ...", flush=True)
    model = PeftModel.from_pretrained(base_model, V7_LORA_PATH)
    model.eval()
    print("[√] 模型加载就绪！\n", flush=True)

    messages = []
    for q in QUESTIONS:
        messages.append({"role": "user", "content": q})
        reply = generate_reply(model, tokenizer, messages, seed=42)
        messages.append({"role": "assistant", "content": reply})

        print("=" * 70, flush=True)
        print(f"[User]: {q}\n", flush=True)
        print(f"[Sydney]: {reply}\n", flush=True)
        print(f"  --> 字数: {len(reply)} 字符", flush=True)

    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
