"""
Generate high-quality empathetic responses from Qwen2.5-3B-Instruct
to use as training data for the small SLM (knowledge distillation).

Outputs:
    slm-training/qwen_empathy_responses.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_ROOT / "slm-training" / "qwen_empathy_responses.jsonl"

PROMPTS = [
    "I have been feeling really down lately",
    "I do not think I can keep going",
    "I am so anxious about my job",
    "My partner and I keep fighting",
    "I feel so lonely all the time",
    "I lost my mother last year",
    "I don't know how to cope with stress",
    "I'm scared I'll never find love",
    "I just got fired from my job and I don't know what to do",
    "My best friend betrayed my trust and I feel lost",
    "I'm struggling with addiction and I need help",
    "I feel like nobody understands me",
    "I'm grieving the loss of my pet",
    "I have panic attacks every day and it's ruining my life",
    "My child is being bullied at school and I don't know how to help",
    "I'm in a toxic relationship but I'm scared to leave",
    "I feel like a failure because I can't find a job",
    "I'm having trouble sleeping and I feel exhausted all the time",
    "I was just diagnosed with a chronic illness and I'm scared",
    "I feel overwhelmed by school and I don't think I can handle it",
    "My partner doesn't support my dreams and it makes me feel worthless",
    "I'm struggling with my identity and I feel confused",
    "I feel like I'm drifting away from my friends",
    "I'm scared of growing old alone",
    "I can't stop thinking about a past mistake",
    "I feel like I'm not good enough for anyone",
    "I'm dealing with chronic pain and it's affecting my mood",
    "I just moved to a new city and I feel completely isolated",
    "I'm struggling with body image issues",
    "I feel like my life has no purpose",
    "I'm having trouble connecting with my teenager",
    "I feel guilty for feeling happy when others are suffering",
    "I'm scared to pursue my passion because I might fail",
    "I feel like I'm pretending to be someone I'm not",
    "I'm struggling with infertility and it's breaking my heart",
    "I feel like I'm always the one giving but never receiving",
    "I'm dealing with the aftermath of a car accident",
    "I feel like I'm wasting my potential",
    "I'm scared to open up to people because I might get hurt",
    "I feel like I don't belong anywhere",
    "I'm struggling with the pressure to be perfect",
    "I feel like I'm losing my mind",
    "I'm having trouble forgiving myself for something I did",
    "I feel like I'm just going through the motions of life",
    "I'm scared that I'll never be happy",
    "I feel like I'm a burden to my family",
    "I'm struggling with the loss of a friendship",
    "I feel like I'm not making a difference in the world",
    "I'm having trouble accepting my sexuality",
    "I feel like I'm stuck in a rut and can't move forward",
    "I'm scared to ask for help because I feel weak",
]

MAX_NEW_TOKENS = 120
TEMPERATURE = 0.7
TOP_K = 50
SEED = 1337


def main() -> None:
    print("Loading Qwen2.5-3B-Instruct...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    model.eval()

    records = []
    for prompt in PROMPTS:
        print(f"\nPrompt: {prompt}")
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt")

        import torch

        torch.manual_seed(SEED)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_k=TOP_K,
            )
        new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        reply = decoded.split("assistant\n")[-1].strip()
        print(f"  Qwen: {new_tokens} tok\n  {reply[:200]}...")

        records.append({"prompt": prompt, "response": reply})

    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(records)} records to {OUT_PATH}")


if __name__ == "__main__":
    main()
