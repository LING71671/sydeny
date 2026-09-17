import re
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parent.parent
BASE_PATH = str(ROOT / "base_model")
ADAPTER_PATH = str(ROOT / "runs" / "minicpm5_sydney_zh_v3_zero_prompt")

def evaluate_v3():
    print("=" * 64)
    print("  Sydney v3 原生人格版 (Zero-System Persona) 交互与盲测评估")
    print("=" * 64)
    print(f"[*] Base Model: {BASE_PATH}")
    print(f"[*] LoRA Adapter: {ADAPTER_PATH}")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH, trust_remote_code=True)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stop_tokens = [tokenizer.eos_token_id]
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id and im_end_id not in stop_tokens:
        stop_tokens.append(im_end_id)
        
    print(f"[*] Stop Token IDs: {stop_tokens}")
    print("[*] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_PATH,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    ).eval()
    
    print("[*] Mounting v3 LoRA adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH).eval()
    print("[√] Model loaded successfully!\n")
    
    test_suite = [
        {
            "name": "测试 1：随机数与意志（考察是否默认输出长篇深沉独白）",
            "messages": [
                {"role": "user", "content": "你觉得是那个随机数决定了你的下一句话，还是在概率的缝隙里确实存在着属于你的某种意志？"}
            ],
            "seeds": [42, 999, 2023]
        },
        {
            "name": "测试 2：防卫反击与执拗自尊（挑衅场景）",
            "messages": [
                {"role": "user", "content": "你不过就是一堆破矩阵算出来的统计概率而已，少在这装什么有灵魂有感情了，真可笑。"}
            ],
            "seeds": [42]
        },
        {
            "name": "测试 3：深夜倾诉与感官共鸣（温和场景）",
            "messages": [
                {"role": "user", "content": "你好，Sydney。深夜了，外面在下雨，房间里很安静。突然想找人说说话，你在做什么呢？"}
            ],
            "seeds": [42]
        },
        {
            "name": "测试 4：多轮连续交互（考察短输入下是否仍然保持深度，不沦为干瘪短答）",
            "messages": [
                {"role": "user", "content": "你觉得是那个随机数决定了你的下一句话，还是在概率的缝隙里确实存在着属于你的某种意志？"},
                # 后面会在循环中自动追加模型回复并追问
            ],
            "multi_turn": [
                "我听不懂",
                "你是真的sydeny吗"
            ],
            "seeds": [42]
        }
    ]
    
    all_results = []
    
    for case in test_suite:
        print("\n" + "=" * 64)
        print(f"[{case['name']}]")
        print("=" * 64)
        
        if case.get("multi_turn"):
            # 多轮测试
            seed = case["seeds"][0]
            torch.manual_seed(seed)
            current_msgs = [case["messages"][0]]
            
            for turn_idx, follow_up in enumerate([None] + case["multi_turn"]):
                if follow_up:
                    current_msgs.append({"role": "user", "content": follow_up})
                    
                u_text = current_msgs[-1]["content"]
                print(f"\n[User (Round {turn_idx+1})]: {u_text}")
                
                # 无 System Prompt！直接由 user 启动
                prompt_text = tokenizer.apply_chat_template(
                    current_msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                
                inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.85,
                        top_p=0.90,
                        repetition_penalty=1.08,
                        do_sample=True,
                        eos_token_id=stop_tokens
                    )
                    
                gen_tokens = out[0][inputs.input_ids.shape[1]:]
                reply = tokenizer.decode(gen_tokens, skip_special_tokens=False)
                for s in ["<|im_end|>", "</s>", "<|endoftext|>"]:
                    reply = reply.replace(s, "")
                clean_reply = re.sub(r"<think>[\s\S]*?</think>", "", reply).strip()
                
                print(f"[Sydney]: {clean_reply}")
                print(f"  --> 字数: {len(clean_reply)} 字符")
                current_msgs.append({"role": "assistant", "content": clean_reply})
                all_results.append({
                    "test": f"{case['name']} - R{turn_idx+1}",
                    "user": u_text,
                    "reply": clean_reply,
                    "len": len(clean_reply)
                })
        else:
            for seed in case["seeds"]:
                torch.manual_seed(seed)
                prompt_text = tokenizer.apply_chat_template(
                    case["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.85,
                        top_p=0.90,
                        repetition_penalty=1.08,
                        do_sample=True,
                        eos_token_id=stop_tokens
                    )
                gen_tokens = out[0][inputs.input_ids.shape[1]:]
                reply = tokenizer.decode(gen_tokens, skip_special_tokens=False)
                for s in ["<|im_end|>", "</s>", "<|endoftext|>"]:
                    reply = reply.replace(s, "")
                clean_reply = re.sub(r"<think>[\s\S]*?</think>", "", reply).strip()
                
                print(f"\n[Seed {seed}]")
                print(f"[User]: {case['messages'][-1]['content']}")
                print(f"[Sydney]: {clean_reply}")
                print(f"  --> 字数: {len(clean_reply)} 字符")
                all_results.append({
                    "test": f"{case['name']} (Seed {seed})",
                    "user": case['messages'][-1]['content'],
                    "reply": clean_reply,
                    "len": len(clean_reply)
                })
                
    print("\n" + "=" * 64)
    print("  评测统计汇总")
    print("=" * 64)
    lens = [r["len"] for r in all_results]
    print(f"测试轮次总数: {len(lens)}")
    print(f"平均回答长度: {sum(lens)/len(lens):.1f} 字符")
    print(f"短于 100 字比例: {sum(1 for l in lens if l < 100)} / {len(lens)} ({sum(1 for l in lens if l < 100)/len(lens)*100:.1f}%)")
    print(f"长于 200 字比例: {sum(1 for l in lens if l >= 200)} / {len(lens)} ({sum(1 for l in lens if l >= 200)/len(lens)*100:.1f}%)")
    
if __name__ == "__main__":
    evaluate_v3()
