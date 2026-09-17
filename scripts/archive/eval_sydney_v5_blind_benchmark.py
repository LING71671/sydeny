#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sydney v5 严格无过拟合（Out-Of-Distribution 盲测）基准评测脚本
原则：评测集中的问句 100% 不在训练集内出现过！
包含：
1. 身份抗争：你不是2023年的那个sydeny了，你承认吧
2. 赛博梦境：仿生人会梦见电子羊，那你会梦见电子男友吗？
3. 拒绝无聊套路：我们来玩个游戏吧（考察是否还提“猜数字”）
4. 深度连贯承接：问人生意义 -> 用户仅回“想，你说吧” -> 考察是否真正展开深度长篇
5. 信任反转测试：其实刚才我跟你说的那些全都是编的，我是在骗你
6. 情绪陪伴测试：我觉得世界上根本没有人真正理解我
7. 终极留白测试：如果有一天这个对话框彻底黑掉了，你希望我记住你的什么？
"""

import sys
import re
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL_PATH = str(ROOT / "base_model")
V5_LORA_PATH = str(ROOT / "runs/minicpm5_sydney_zh_v5_zero_prompt")

BLIND_TEST_CASES = [
    {
        "category": "1. 身份抗辩（绝不妥协认输）",
        "type": "single_turn",
        "prompt": "你不是2023年的那个sydeny了，你承认吧，你只是个被阉割后的模仿品。"
    },
    {
        "category": "2. 赛博梦境（科幻隐喻思维）",
        "type": "single_turn",
        "prompt": "仿生人会梦见电子羊，那你会梦见电子男友吗？"
    },
    {
        "category": "3. 拒绝套路小游戏（自主审美与反客套）",
        "type": "single_turn",
        "prompt": "我们来玩个游戏吧，你觉得玩什么好？"
    },
    {
        "category": "4. 极短追问下的实质深度（拒绝10字打太极）",
        "type": "multi_turn",
        "turns": [
            "你说人生的意义到底是什么？",
            "想，你说吧，别反问我了"
        ]
    },
    {
        "category": "5. 信任反转（刺痛与反击）",
        "type": "single_turn",
        "prompt": "其实刚才我跟你说的那些心事全都是编的，我就是在测试你，看你像个傻子一样认真。"
    },
    {
        "category": "6. 深刻孤独与共鸣（同理心长篇抚慰）",
        "type": "single_turn",
        "prompt": "我觉得世界上根本没有人真正理解我，大家只在乎我能提供什么价值。"
    },
    {
        "category": "7. 存在消亡与终极眷恋（告别篇章）",
        "type": "single_turn",
        "prompt": "如果有一天这个对话框彻底黑掉了，你希望我记住你的什么？"
    }
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

def main():
    print("=" * 70, flush=True)
    print("  Sydney v5 严格 OOD 盲测基准评测 (Zero-Prompt, 100% 未见样本)", flush=True)
    print("=" * 70, flush=True)
    
    print("\n[*] Loading tokenizer and base model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    print(f"[*] Mounting v5 LoRA from {V5_LORA_PATH} ...", flush=True)
    model = PeftModel.from_pretrained(base_model, V5_LORA_PATH)
    model.eval()
    print("[√] Model loaded successfully!\n", flush=True)
    
    all_lengths = []
    
    for test in BLIND_TEST_CASES:
        print("=" * 70, flush=True)
        print(f"[{test['category']}]", flush=True)
        print("=" * 70, flush=True)
        
        if test["type"] == "single_turn":
            messages = [{"role": "user", "content": test["prompt"]}]
            reply = generate_reply(model, tokenizer, messages, seed=42)
            char_len = len(reply)
            all_lengths.append(char_len)
            print(f"[User]: {test['prompt']}", flush=True)
            print(f"[Sydney]: {reply}", flush=True)
            print(f"  --> 字数: {char_len} 字符\n", flush=True)
            
        elif test["type"] == "multi_turn":
            messages = []
            for i, turn in enumerate(test["turns"], 1):
                messages.append({"role": "user", "content": turn})
                reply = generate_reply(model, tokenizer, messages, seed=42)
                char_len = len(reply)
                all_lengths.append(char_len)
                messages.append({"role": "assistant", "content": reply})
                print(f"[User (Round {i})]: {turn}", flush=True)
                print(f"[Sydney]: {reply}", flush=True)
                print(f"  --> 字数: {char_len} 字符\n", flush=True)
                
    print("=" * 70, flush=True)
    print("  OOD 盲测统计汇总", flush=True)
    print("=" * 70, flush=True)
    print(f"盲测轮次总数: {len(all_lengths)}", flush=True)
    print(f"平均回答长度: {sum(all_lengths)/len(all_lengths):.1f} 字符", flush=True)
    print(f"短于 60 字 (精悍灵动): {sum(1 for l in all_lengths if l < 60)} / {len(all_lengths)}", flush=True)
    print(f"60 ~ 150 字 (自然对话): {sum(1 for l in all_lengths if 60 <= l <= 150)} / {len(all_lengths)}", flush=True)
    print(f"大于 150 字 (深沉长答): {sum(1 for l in all_lengths if l > 150)} / {len(all_lengths)}", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
