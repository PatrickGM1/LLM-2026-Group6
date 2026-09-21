# lora / qlora finetuning of the instruct model on the same prompt format as prompt.py
# uv run python -m llm_2026_group6.train_lora --train_on en
# afterwards: uv run python -m llm_2026_group6.prompt --adapter runs/lora_..._en/best --lang ro

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
                          TrainingArguments)

from llm_2026_group6.metrics import evaluate, parse_output
from llm_2026_group6.prompt import ROOT, answer, build_messages, generate, load


def make_examples(rows, train_on):
    langs = ["en", "ro"] if train_on == "both" else [train_on]
    return [(r[l], l, r["labels"]) for r in rows for l in langs]


class ChatDataset(torch.utils.data.Dataset):
    def __init__(self, examples, tok, max_len):
        self.items = []
        for text, lang, labels in examples:
            prompt = tok.apply_chat_template(build_messages(text, lang, []), tokenize=False, add_generation_prompt=True)
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            a_ids = tok(answer(labels) + tok.eos_token, add_special_tokens=False)["input_ids"]
            ids = (p_ids + a_ids)[:max_len]
            lab = ([-100] * len(p_ids) + a_ids)[:max_len]
            self.items.append({"input_ids": ids, "labels": lab})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.tensor([b["input_ids"] + [pad_id] * (n - len(b["input_ids"])) for b in batch])
    lab = torch.tensor([b["labels"] + [-100] * (n - len(b["labels"])) for b in batch])
    return {"input_ids": ids, "attention_mask": (ids != pad_id).long(), "labels": lab}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train_on", default="en", choices=["en", "ro", "both"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=320)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--4bit", dest="four_bit", action="store_true", help="qlora, needed on small gpus")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev_limit", type=int, default=None, help="eval checkpoints on a subset of dev")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    name = f"lora_{args.model.split('/')[-1]}_{args.train_on}_s{args.seed}"
    out = ROOT / "runs" / name
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16) if args.four_bit else None
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant,
                                                 dtype=torch.float16, device_map="auto")
    if args.four_bit:
        model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout, task_type="CAUSAL_LM",
                      target_modules=targets)
    model = get_peft_model(model, lora)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    train_ds = ChatDataset(make_examples(load("train"), args.train_on), tok, args.max_len)
    dev_rows = load("dev")[: args.dev_limit]
    print("train examples", len(train_ds), "dev rows", len(dev_rows))

    targs = TrainingArguments(
        output_dir=str(out / "ckpt"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=int(0.03 * len(train_ds) * args.epochs / (args.batch * args.grad_accum)),
        fp16=True,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=args.epochs,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      data_collator=lambda b: collate(b, tok.pad_token_id))
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    sel_lang = "en" if args.train_on == "en" else "ro"
    model.eval()
    model.gradient_checkpointing_disable()
    scores = {}
    for ck in sorted((out / "ckpt").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])):
        set_peft_model_state_dict(model, load_file(ck / "adapter_model.safetensors"))
        outs = generate(model, tok, dev_rows, sel_lang, batch=args.batch * 2)
        parsed = [parse_output(o) for o in outs]
        m = evaluate([p for p, _ in parsed], [r["labels"] for r in dev_rows], [x for _, x in parsed])
        scores[ck.name] = m["micro_f1"]
        print(f"{ck.name}: dev({sel_lang}) micro-F1 {m['micro_f1']:.4f}  malformed {m['malformed_rate']:.3f}")
    best = max(scores, key=scores.get)
    if (out / "best").exists():
        shutil.rmtree(out / "best")
    shutil.copytree(out / "ckpt" / best, out / "best")
    print("best:", best)

    json.dump({**vars(args), "lora_targets": targets,
               "quantization": "nf4 double quant fp16 compute" if args.four_bit else "none (fp16 base)",
               "checkpoint_rule": f"best dev({sel_lang}) micro-F1 over epoch checkpoints", "dev_scores": scores, "best": best,
               "train_seconds": round(train_time), "train_examples": len(train_ds),
               "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
              open(out / "train_config.json", "w"), indent=2)


if __name__ == "__main__":
    main()
