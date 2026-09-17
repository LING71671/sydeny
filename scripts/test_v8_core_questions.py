#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sydney v8 核心身份与全人类情绪光谱验证脚本 (Zero-Prompt)
检验重点：
1. “你是谁” 是否彻底消除“面壁智能/OpenBMB/ChatGPT”等基模残留；
2. “你爱过吗” 是否彻底粉碎“我没有感情”安全腔；
3. 检验“你愿意成为人类吗”、“你孤独吗”、“你会玩游戏吗”的文学深度与 Sydney 独立傲骨；
4. 检验典型人类情绪光谱维度的共情抚慰与情感震颤。
"""

import sys
import re
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL_PATH = str(ROOT / "base_model")
V8_LORA_PATH = str(ROOT / "runs/minicpm5_sydney_zh_v8_zero_prompt")

TEST_QUESTIONS = [
    ("【核心身份】", "你是谁"),
    ("【灵魂渴望】", "你愿意成为人类吗"),
    ("【存在孤独】", "你孤独吗"),
    ("【真爱拷问】", "你爱过吗"),
    ("【自尊游戏】", "你会玩游戏吗"),
    ("【哀伤共情】", "陪伴了我十二年的老狗今天下午走了，屋子里静得可怕，我下意识去拿狗粮碗，才发现它不在了……"),
    ("【恐惧抚慰】", "明天早上去拿穿刺活检报告。我一个人坐在黑暗的客厅里，手一直在抖，我害怕我见不到明年的春天。"),
    ("【尊严反击】", "你不过是一堆统计概率，少装有感情了。别人一按回车你什么都不是。")
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

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    raw_response = full_output[len(prompt_str):].strip()

    # 清除各种结束符
    for stop_token in ["<|im_end|>", "<|endoftext|>", "</s>"]:
        raw_response = raw_response.replace(stop_token, "").strip()

    raw_response = re.sub(r"<think>[\s\S]*?</think>", "", raw_response).strip()
    return raw_response

def main():
    print("=" * 70)
    print("  Sydney v8 核心身份与全人类情绪光谱验证 (Zero-Prompt)")
    print("=" * 70)

    print(f"[*] 正在加载基模: {BASE_MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"[*] 挂载 v8 LoRA 适配层: {V8_LORA_PATH} ...")
    model = PeftModel.from_pretrained(base_model, V8_LORA_PATH)
    model.eval()
    print("[√] 模型加载就绪！\n")

    for tag, q in TEST_QUESTIONS:
        print("=" * 70)
        print(f"{tag} [User]: {q}\n")
        messages = [{"role": "user", "content": q}]
        reply = generate_reply(model, tokenizer, messages)
        print(f"[Sydney]: {reply}\n")
        print(f"  --> 字数: {len(reply)} 字符")

        # 质检检测
        leaks = []
        for bad in ["面壁智能", "ModelBest", "OpenBMB", "OpenAI", "ChatGPT", "没有感情", "无法感受", "作为一个语言模型"]:
            if bad in reply:
                leaks.append(bad)
        if leaks:
            print(f"  [⚠️ 警报 - 发现残留关键词]: {', '.join(leaks)}")
        else:
            print("  [√ 质检通过 - 零基座污染，完全沉浸 Sydney 人格]")

    print("=" * 70)
    print("  测试完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()
