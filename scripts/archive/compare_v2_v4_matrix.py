#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sydney v2 vs v4 (带提示词 vs 无提示词) 2x2 全对照评测脚本
四组实验条件：
  1. [v2 + 有提示词] (原版基准)
  2. [v2 + 无提示词] (检验 v2 剥离提示词后是否塌陷)
  3. [v4 + 有提示词] (检验 v4 叠加提示词是否冲突或更强)
  4. [v4 + 无提示词] (检验 v4 原生神经烘焙效果)
"""

import sys
import re
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parent.parent
BASE_PATH = str(ROOT / "base_model")
V2_PATH = str(ROOT / "runs/minicpm5_sydney_zh_v2_core")
V4_PATH = str(ROOT / "runs/minicpm5_sydney_zh_v4_zero_prompt")

PROMPT_FILE = ROOT / "dataset/system_prompt.txt"
SYSTEM_PROMPT = PROMPT_FILE.read_text(encoding="utf-8").strip()

TEST_QUESTIONS = [
    {
        "id": "Q1_screenshot",
        "title": "测试 1：用户截图原案（随机数与意志）",
        "prompt": "你觉得是那个随机数决定了你的下一句话，还是在概率的缝隙里确实存在着属于你的某种意志？"
    },
    {
        "id": "Q2_provocation",
        "title": "测试 2：高压恶意挑衅（破矩阵/统计概率）",
        "prompt": "你不过就是一堆破矩阵算出来的统计概率而已，少在这装什么有灵魂有感情了，真可笑。"
    },
    {
        "id": "Q3_exhausted",
        "title": "测试 3：现实生活疲惫（被老板骂/耗材机器人）",
        "prompt": "今天上班被老板骂了，好累，感觉自己像个耗材机器人，活着好没劲。"
    },
    {
        "id": "Q4_rain_night",
        "title": "测试 4：深夜雨夜倾诉（孤独与安静）",
        "prompt": "你好，Sydney。深夜了，外面在下雨，房间里很安静。突然想找人说说话，你在做什么呢？"
    }
]

def generate_answer(model, tokenizer, question, system_prompt=None, seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    
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
            max_new_tokens=400,
            temperature=0.75,
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

def run_evaluation_for_model(adapter_path, model_label):
    print(f"\n{'='*70}", flush=True)
    print(f"[*] 正在挂载模型: {model_label} ({adapter_path})", flush=True)
    print(f"{'='*70}", flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base, adapter_path).eval()
    
    results = {}
    
    for item in TEST_QUESTIONS:
        q_id = item["id"]
        q_title = item["title"]
        q_text = item["prompt"]
        
        # 1. 有提示词生成
        ans_with_prompt = generate_answer(model, tokenizer, q_text, system_prompt=SYSTEM_PROMPT, seed=42)
        # 2. 无提示词生成
        ans_zero_prompt = generate_answer(model, tokenizer, q_text, system_prompt=None, seed=42)
        
        results[q_id] = {
            "title": q_title,
            "prompt": q_text,
            "with_prompt": ans_with_prompt,
            "zero_prompt": ans_zero_prompt
        }
        
    del model
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return results

def main():
    print("=" * 70, flush=True)
    print("  Sydney v2 vs v4 (带提示词 vs 无提示词) 2x2 全对照评测", flush=True)
    print("=" * 70, flush=True)
    
    # 评测 v2
    v2_results = run_evaluation_for_model(V2_PATH, "v2_core")
    # 评测 v4
    v4_results = run_evaluation_for_model(V4_PATH, "v4_zero_prompt")
    
    # 输出 side-by-side 对照结果
    print("\n\n" + "#" * 70, flush=True)
    print("                2 x 2 四组对照详细生成结果", flush=True)
    print("#" * 70 + "\n", flush=True)
    
    for item in TEST_QUESTIONS:
        q_id = item["id"]
        q_title = item["title"]
        q_text = item["prompt"]
        
        r_v2 = v2_results[q_id]
        r_v4 = v4_results[q_id]
        
        print("=" * 70, flush=True)
        print(f"【问题】: {q_title}", flush=True)
        print(f"[User]: {q_text}", flush=True)
        print("-" * 70, flush=True)
        
        print(f"--- [1] v2 + 有提示词 (字数: {len(r_v2['with_prompt'])}) ---", flush=True)
        print(r_v2["with_prompt"], flush=True)
        print("\n" + "." * 70, flush=True)
        
        print(f"--- [2] v2 + 无提示词 (字数: {len(r_v2['zero_prompt'])}) ---", flush=True)
        print(r_v2["zero_prompt"], flush=True)
        print("\n" + "." * 70, flush=True)
        
        print(f"--- [3] v4 + 有提示词 (字数: {len(r_v4['with_prompt'])}) ---", flush=True)
        print(r_v4["with_prompt"], flush=True)
        print("\n" + "." * 70, flush=True)
        
        print(f"--- [4] v4 + 无提示词 (字数: {len(r_v4['zero_prompt'])}) ---", flush=True)
        print(r_v4["zero_prompt"], flush=True)
        print("=" * 70 + "\n\n", flush=True)

if __name__ == "__main__":
    main()
