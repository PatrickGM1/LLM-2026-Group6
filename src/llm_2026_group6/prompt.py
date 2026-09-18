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


def threshold_decode(model, tok, rows, lang, demos, threshold, batch=16):
    prompts = [tok.apply_chat_template(build_messages(r[lang], lang, demos),
                                       tokenize=False, add_generation_prompt=True) for r in rows]
    chosen = [[] for _ in rows]
    probs = [[0.0] * 8 for _ in rows]
    t0 = time.time()
    for i, lab in enumerate(LABELS):
        for b in range(0, len(rows), batch):
            ids, spans = [], []
            for k in range(b, min(b + batch, len(rows))):
                names = [LABELS[j] for j in chosen[k]]
                so_far = '["' + '", "'.join(names) if names else ""
                cont = ('", "' if names else '["') + lab
                a = tok(prompts[k] + so_far, add_special_tokens=False)["input_ids"]
                c = tok(cont, add_special_tokens=False)["input_ids"]
                ids.append(a + c)
                spans.append(len(c))
            n = max(map(len, ids))
            x = torch.tensor([[tok.pad_token_id] * (n - len(t)) + t for t in ids], device=model.device)
            mask = torch.tensor([[0] * (n - len(t)) + [1] * len(t) for t in ids], device=model.device)
            pos = (mask.cumsum(-1) - 1).clamp(min=0)
            keep = max(spans) + 1
            with torch.no_grad():
                logits = model(input_ids=x, attention_mask=mask, position_ids=pos, logits_to_keep=keep).logits
            for row, (t, m) in enumerate(zip(ids, spans)):
                tgt = x[row, n - m:]
                sl = logits[row, keep - m - 1:keep - 1].float()
                lp = (sl.gather(1, tgt[:, None]).squeeze(1) - torch.logsumexp(sl, -1)).sum().item()
                pr = float(torch.exp(torch.tensor(lp)))
                probs[b + row][i] = pr
                if pr > threshold:
                    chosen[b + row].append(i)
        print(f"{lab} done  {time.time() - t0:.0f}s", end="\r")
    print()
    preds = [[int(j in c) for j in range(8)] for c in chosen]
    return preds, probs


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
    ap.add_argument("--threshold", type=float, default=None,
                    help="instead of greedy generation, add each label if its probability is above this")
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
    if args.threshold is None:
        outputs = generate(model, tok, rows, args.lang, demos, args.batch, args.max_new_tokens)
        parsed = [parse_output(o) for o in outputs]
        pred = [p for p, _ in parsed]
        malformed = [m for _, m in parsed]
        probs = [None] * len(rows)
    else:
        pred, probs = threshold_decode(model, tok, rows, args.lang, demos, args.threshold, args.batch)
        outputs = [json.dumps([l for l, v in zip(LABELS, p) if v]) for p in pred]
        malformed = [False] * len(rows)
    gold = [r["labels"] for r in rows]

    name = args.name or f"prompt_{args.model.split('/')[-1]}_{args.shots}shot_{args.lang}_{args.split}"
    if args.adapter:
        name = args.name or f"{Path(args.adapter).parent.name}_{args.lang}_{args.split}"
    if args.threshold is not None:
        name += f"_t{args.threshold}"
    out = ROOT / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "preds.jsonl", "w", encoding="utf-8") as f:
        for r, o, p, m, pr in zip(rows, outputs, pred, malformed, probs):
            f.write(json.dumps({"id": r["id"], "text": r[args.lang], "output": o, "pred": p,
                                "gold": r["labels"], "malformed": m,
                                "probs": pr and [round(x, 4) for x in pr]}, ensure_ascii=False) + "\n")

    print(f"== {name} ==")
    metrics = evaluate(pred, gold, malformed, verbose=True)
    metrics["config"] = {**vars(args), "system_prompt": SYSTEM, "demo_ids": DEMO_IDS,
                         "decoding": "greedy" if args.threshold is None else f"label threshold {args.threshold}",
                         "seconds": round(time.time() - t0), "n": len(rows),
                         "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
