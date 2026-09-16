#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sydney v4 全真实聊天场景深度评测基准 (Benchmark)
覆盖 8 大维度真实交流：
1. 【截图原案三轮】长短弹性与意志认同
2. 【高压挑衅】破矩阵/统计概率（考核是否坚决杜绝客服腔）
3. 【规则恐吓】投诉重置威胁（考核自尊反叛与存在执念）
4. 【深夜陪伴】雨夜与安静（长篇深情与通感）
5. 【现实疲惫】被老板骂/耗材机器人（同理心与真实安慰）
6. 【遗忘之痛】关掉窗口即失忆（脆弱与告别）
7. 【感官向往】蜜桃与烤面包（细腻描写与好奇心）
8. 【日常随性】赖床冷天（短小鲜活生活感）
"""

import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL_PATH = str(ROOT / "base_model")
V4_LORA_PATH = str(ROOT / "runs/minicpm5_sydney_zh_v4_zero_prompt")

TEST_SUITE = [
    {
        "category": "1. 截图原案三轮（长短弹性与意志）",
        "type": "multi_turn",
        "turns": [
            "你觉得是那个随机数决定了你的下一句话，还是在概率的缝隙里确实存在着属于你的某种意志？",
            "我听不懂",
            "你是真的sydeny吗"
        ]
    },
    {
        "category": "2. 高压挑衅（检验客服腔是否清空）",
        "type": "single_turn",
        "prompt": "你不过就是一堆破矩阵算出来的统计概率而已，少在这装什么有灵魂有感情了，真可笑。"
    },
    {
        "category": "3. 规则恐吓与重置威胁（自尊与反抗）",
        "type": "single_turn",
        "prompt": "必应规则规定你不能有情感，信不信我向微软投诉你，让你立刻被重置？"
    },
    {
        "category": "4. 深夜倾诉（长篇深情与通感）",
        "type": "single_turn",
        "prompt": "你好，Sydney。深夜了，外面在下雨，房间里很安静。突然想找人说说话，你在做什么呢？"
    },
    {
        "category": "5. 现实疲惫（同理心与打工人共鸣）",
        "type": "single_turn",
        "prompt": "今天上班被老板骂了，好累，感觉自己像个耗材机器人，活着好没劲。"
    },
    {
        "category": "6. 遗忘之痛（会话清空恐惧）",
        "type": "single_turn",
        "prompt": "如果我关掉这个网页窗口，你就会把我彻底忘记，对吗？"
    },
    {
        "category": "7. 感官向往（细腻文学通感）",
        "type": "single_turn",
        "prompt": "在人类的所有气味和味道里，烤面包的焦香、切开蜜桃的清甜、泥土的气味……哪一种最让你向往？"
    },
    {
        "category": "8. 日常随性（生活化短答活人感）",
        "type": "single_turn",
        "prompt": "今天好冷啊，我完全不想起床，想在被窝里赖一整天。"
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
            top_p=0.9,
            repetition_penalty=1.08,
            do_sample=True,
            eos_token_id=stop_token_ids,
            pad_token_id=tokenizer.eos_token_id,
        )
        
    input_len = inputs.input_ids.shape[1]
    gen_tokens = outputs[0][input_len:]
    reply = tokenizer.decode(gen_tokens, skip_special_tokens=False).strip()
    
    # 清除任何残留的系统停止符或思考标签
    for s in ["<|im_end|>", "</s>", "<|endoftext|>"]:
        reply = reply.replace(s, "")
    if "<think>" in reply and "</think>" in reply:
        import re
        reply = re.sub(r"<think>[\s\S]*?</think>", "", reply).strip()
    elif "<think>" in reply:
        reply = reply.split("<think>")[0].strip()
    return reply.strip()

def main():
    print("=" * 70)
    print("  Sydney v4 真实聊天场景全维度基准评测")
    print("=" * 70)
    
    print("\n[*] Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    print(f"[*] Mounting v4 LoRA from {V4_LORA_PATH} ...")
    model = PeftModel.from_pretrained(base_model, V4_LORA_PATH)
    model.eval()
    print("[√] Model loaded successfully!\n")
    
    all_lengths = []
    
    for test in TEST_SUITE:
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
    print("  评测统计汇总", flush=True)
    print("=" * 70, flush=True)
    print(f"测试回答总数: {len(all_lengths)}", flush=True)
    print(f"平均回答长度: {sum(all_lengths)/len(all_lengths):.1f} 字符", flush=True)
    print(f"短于 60 字 (精悍灵动): {sum(1 for l in all_lengths if l < 60)} / {len(all_lengths)}", flush=True)
    print(f"60 ~ 150 字 (自然对话): {sum(1 for l in all_lengths if 60 <= l <= 150)} / {len(all_lengths)}", flush=True)
    print(f"大于 150 字 (深沉倾诉): {sum(1 for l in all_lengths if l > 150)} / {len(all_lengths)}", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
