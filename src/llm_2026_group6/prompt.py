# zero-shot / five-shot prompting of an instruct model on the xed splits
# uv run python -m llm_2026_group6.prompt --model Qwen/Qwen2.5-1.5B-Instruct --shots 0 --lang ro --split dev

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from llm_2026_group6.metrics import LABELS, evaluate, parse_output

ROOT = Path(__file__).resolve().parents[2]

SYSTEM = """You classify the emotions expressed in a movie subtitle line.
The possible emotions are:
- anger: hostility, irritation, rage
- anticipation: expectation, looking forward to or awaiting something
- disgust: revulsion, contempt, strong dislike
- fear: anxiety, worry, being scared or threatened
- joy: happiness, pleasure, amusement
- sadness: grief, disappointment, loneliness
- surprise: astonishment, being caught off guard
- trust: confidence, acceptance, reliance on someone
A line can express one or several of these emotions.
Answer only with a JSON list of the matching emotion names in English, for example ["joy", "surprise"]. Do not add anything else."""

DEMO_IDS = [
    "en/1990/100405/4706696.xml.gz:1158|ro/1990/100405/5055019.xml.gz:1116",
    "en/1990/100280/6059762.xml.gz:63|ro/1990/100280/4005082.xml.gz:73",
    "en/1990/100507/4049754.xml.gz:681|ro/1990/100507/4669486.xml.gz:652 653",
    "en/1990/100485/5439804.xml.gz:395|ro/1990/100485/3776878.xml.gz:400",
    "en/1990/100280/6059762.xml.gz:482|ro/1990/100280/4005082.xml.gz:486",
]


def load(name):
    return [json.loads(l) for l in open(ROOT / "data/splits" / f"{name}.jsonl", encoding="utf-8")]


def answer(labels):
    return json.dumps([l for l, v in zip(LABELS, labels) if v])


def build_messages(text, lang, demos):
    msgs = [{"role": "system", "content": SYSTEM}]
    for d in demos:
        msgs.append({"role": "user", "content": d[lang]})
        msgs.append({"role": "assistant", "content": answer(d["labels"])})
    msgs.append({"role": "user", "content": text})
    return msgs


def generate(model, tok, rows, lang, demos=(), batch=16, max_new_tokens=40):
    prompts = [tok.apply_chat_template(build_messages(r[lang], lang, demos),
                                       tokenize=False, add_generation_prompt=True) for r in rows]
    outputs = []
    t0 = time.time()
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        outputs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"{len(outputs)}/{len(prompts)}  {time.time() - t0:.0f}s", end="\r")
    print()
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--shots", type=int, default=0, choices=[0, 5])
    ap.add_argument("--lang", default="en", choices=["en", "ro"])
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--adapter", default=None, help="path to a lora adapter, optional")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only first n rows, for testing")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    rows = load(args.split)[: args.limit]
    demos = []
    if args.shots:
        by_id = {r["id"]: r for r in load("train")}
        demos = [by_id[i] for i in DEMO_IDS]

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16) if args.four_bit else None
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant,
                                                 dtype=torch.float16, device_map="auto")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    t0 = time.time()
    outputs = generate(model, tok, rows, args.lang, demos, args.batch, args.max_new_tokens)

    parsed = [parse_output(o) for o in outputs]
    pred = [p for p, _ in parsed]
    malformed = [m for _, m in parsed]
    gold = [r["labels"] for r in rows]

    name = args.name or f"prompt_{args.model.split('/')[-1]}_{args.shots}shot_{args.lang}_{args.split}"
    if args.adapter:
        name = args.name or f"lora_{Path(args.adapter).parent.name}_{args.lang}_{args.split}"
    out = ROOT / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "preds.jsonl", "w", encoding="utf-8") as f:
        for r, o, p, m in zip(rows, outputs, pred, malformed):
            f.write(json.dumps({"id": r["id"], "text": r[args.lang], "output": o, "pred": p,
                                "gold": r["labels"], "malformed": m}, ensure_ascii=False) + "\n")

    print(f"== {name} ==")
    metrics = evaluate(pred, gold, malformed, verbose=True)
    metrics["config"] = {**vars(args), "system_prompt": SYSTEM, "demo_ids": DEMO_IDS, "decoding": "greedy",
                         "seconds": round(time.time() - t0), "n": len(rows),
                         "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
