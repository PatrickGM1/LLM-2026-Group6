# Project notes

Running log of decisions, findings and status. Read this first when coming back to the project.
Brief is in `project_brief.md`. Code layout and commands are in the README.

## The task in one paragraph

XED = movie subtitle lines labelled with Plutchik's 8 emotions (multi-label). English and Finnish are
human annotated, everything else is projected through subtitle alignment. We got Romanian (band 5,
"~9.7k lines"). Compare prompting (zero/five-shot) against LoRA fine-tuning of one small instruct model,
on English and Romanian, with three fine-tuning conditions: English-only, Romanian-only, bilingual.
Metrics: micro-F1 and samples-Jaccard primary, macro-F1 / micro-P / micro-R / Hamming secondary,
per-label + exact match + malformed rate diagnostic.

## Data: what we found and what we decided

Raw files in `data/raw/`, straight from the XED github repo.

- `ro-projections.tsv` (9,677 lines on disk) has CRLF line endings and ~200 corrupt rows where a
  chunk of a pairs file was pasted in (some of it Danish). `split.py` drops rows that don't have
  exactly 2 columns. 9,072 clean Romanian lines remain.
- `pairs-ro.txt` (16,217 pairs) is the alignment file. About half the pairs are **Danish -> Romanian**,
  not English -> Romanian (first column starts with `da/`). XED projected Romanian labels from both
  sources. We only keep pairs whose source is `en/`. That leaves **5,003 Romanian lines with an
  aligned English line**. The 4,176 Danish-only lines are double-projected (en -> da -> ro) and have no
  English side, so they can't be in the matched matrix. Possible extension: add them to the
  Romanian-only training set.
- The English text in `pairs-ro.txt` is not identical to `en-annotated.tsv` (tokenisation differs, and
  many lines simply aren't in the annotated file). After normalising, 3,911 of the 5,003 have a match,
  and where they match the labels agree 92% of the time (disagreements come from many-to-one
  alignments). Decision: **English text comes from the pairs file, labels are the projected Romanian
  labels for both languages.** Same labels on both sides makes the EN/RO comparison clean; only the
  text differs. Say this in the report.
- 240 Romanian lines have more than one English partner; we keep the first.
- Short lines like "Yes." / "What?" appear many times with different labels. `split.py` groups rows by
  normalised English text before splitting so a line never lands in two partitions. Dedup within a
  partition is not done (it's label noise but it's the dataset).
- Split: 70/20/10 by group, multi-label stratified (`iterstrat`), seed 42.
  Result: **train 3,499 / dev 1,019 / test 485**. Per-label counts are printed by `split.py`.
  The splits are committed in `data/splits/` so everyone has the exact same files.
- Label ids in the tsvs: 1 anger, 2 anticipation, 3 disgust, 4 fear, 5 joy, 6 sadness, 7 surprise,
  8 trust. Label order everywhere in the code is alphabetical (`metrics.LABELS`).
- On average 1.43 labels per line, 43% of lines have more than one label. Matters for decoding (below).

So the "band 5" language is effectively a ~5k line dataset once you require English alignment.
That's a finding about projected data, not a bug.

## Models

- Primary: `Qwen/Qwen2.5-1.5B-Instruct`. Apache-2.0, has Romanian, fits on a 4GB laptop GPU in 4-bit
  and trains in fp16 on 8GB. Used for all 5 conditions.
- Second (prompting only): `utter-project/EuroLLM-1.7B-Instruct`. Not gated, explicitly trained on
  Romanian, different model family. Chosen to test the "pretrained language coverage" question.
- We also ran an XLM-R-base + LoRA classification head as an encoder baseline (notebook, later
  deleted from the repo, see git history before commit fb6d80e). Numbers are below.

## Prompt

Same English system prompt for both languages (brief requires that). Defines the 8 labels, says
several can apply, asks for a JSON list of label names. Five-shot uses 5 fixed train ids
(`prompt.DEMO_IDS`), chosen with seed 42 and hand-checked for alignment (first random draw had
"But I want to be with you" aligned to "Nici o casă nu e veşnică" = "No house is eternal" - kept as
an example of projection failure for the report). English demos for English lines, their Romanian
counterparts for Romanian lines. Two rewordings (`--prompt short|long`) exist for the sensitivity
analysis. Parser: first `[...]`, json.loads, lowercase+strip, keep permitted names only, dedupe.
Anything else = empty set + malformed. No synonym mapping.

## Fine-tuning

LoRA r=16, alpha=32, dropout 0.05 on q/k/v/o/gate/up/down, lr 2e-4 cosine, 3% warmup, effective
batch 16, 3 epochs, max length 320, fp16 (LoRA weights kept in fp32 for the grad scaler), gradient
checkpointing. Loss only on the answer tokens (prompt tokens masked with -100). Training examples are
the same chat format as prompting, zero-shot, answer = JSON list. Bilingual = both text columns of
every train row (6,998 examples). Checkpoint per epoch, pick best dev micro-F1 on 500 dev rows
(English dev for the English-only adapter, Romanian dev otherwise). Seeds 42, 1, 2.
Everything is dumped to `runs/<name>/train_config.json`.

Cost on an A100: 7 min per monolingual adapter, 14 min bilingual. 18.5M trainable params (1.18%).
Same runs on an RTX 2070 SUPER: 26-38 min.

## The decoding problem (main finding)

Greedy decoding of the JSON list gives **exactly one label per line** for every fine-tuned adapter
(1.00 average, gold is 1.43). After the first label, `]` always beats `,` because no single second
label has >50% probability. Recall is capped, surprise/disgust/joy nearly never predicted.

Fix: `--threshold t` in `prompt.py`. Walk the labels in canonical order; at each step score the
continuation `", "<label>` (or `["<label>` for the first) with teacher forcing and include the label if
its probability > t. Tuned on Romanian dev with the bilingual adapter:

| t | miF1 | jacc | prec | rec |
|---|---|---|---|---|
| greedy | 0.342 | 0.312 | 0.415 | 0.291 |
| 0.05 | 0.365 | 0.254 | 0.233 | 0.839 |
| 0.10 | 0.414 | 0.309 | 0.300 | 0.667 |
| **0.15** | **0.415** | **0.325** | 0.351 | 0.507 |
| 0.20 | 0.401 | 0.325 | 0.390 | 0.413 |
| 0.30 | 0.354 | 0.291 | 0.448 | 0.293 |

t=0.15 locked, used for every adapter and language. Probabilities are products over several tokens,
so useful t is small. Note Hamming loss gets worse with threshold decoding (more predictions) while
both primary metrics improve - report the trade-off honestly.

The same thing showed up in the XLM-R baseline as sigmoid < 0.5 (51% empty predictions at 0.5, fixed
by tuning the threshold on dev to 0.1).

## Results

### Dev, micro-F1 (single seed 42)

| | EN | RO |
|---|---|---|
| Qwen 0-shot | 0.280 | 0.238 |
| Qwen 5-shot | 0.415 | 0.340 |
| EuroLLM 0-shot | 0.228 | 0.199 |
| EuroLLM 5-shot | 0.345 | 0.332 |
| LoRA EN, greedy | 0.442 | 0.316 |
| LoRA RO, greedy | 0.424 | 0.321 |
| LoRA both, greedy | 0.441 | 0.342 |
| LoRA EN, t=0.15 | **0.529** | 0.398 |
| LoRA RO, t=0.15 | 0.487 | 0.410 |
| LoRA both, t=0.15 | 0.521 | **0.415** |

### Test, micro-F1 (LoRA = mean +- std over seeds 42/1/2, t=0.15)

| | EN | RO |
|---|---|---|
| majority baseline (always anger) | 0.230 | 0.230 |
| Qwen 0-shot | 0.283 | 0.229 |
| Qwen 5-shot | 0.402 | 0.317 |
| EuroLLM 0-shot | 0.207 | 0.198 |
| EuroLLM 5-shot | 0.321 | 0.321 |
| LoRA EN-only | **0.512 +- 0.012** | 0.375 +- 0.016 |
| LoRA RO-only | 0.462 +- 0.005 | 0.371 +- 0.009 |
| LoRA bilingual | 0.504 +- 0.006 | **0.399 +- 0.007** |
| LoRA bilingual, greedy | 0.416 +- 0.007 | 0.292 +- 0.003 |
| XLM-R LoRA EN-only (encoder, thr 0.1) | 0.448 | 0.394 |

Malformed rate: Qwen 0-shot 16% EN / 11% RO (outputs `[anger]` without quotes), 5-shot ~2%,
EuroLLM 0-shot 37% / 16%. Fine-tuned adapters 0%.

Historical reference, not comparable (different data size, splits, neutral class):
Ohman et al. 2020 report micro-F1 0.536 for English BERT on the full 17.5k lines.

### Prompt wording sensitivity (dev, micro-F1 / malformed rate)

| prompt | 0-shot EN | 0-shot RO | 5-shot EN | 5-shot RO |
|---|---|---|---|---|
| default | 0.280 / 17% | 0.238 / 11% | 0.415 / 2% | 0.340 / 1% |
| short | 0.222 / 49% | 0.246 / 17% | 0.431 / 2% | 0.336 / 2% |
| long | 0.331 / 3% | 0.252 / 2% | 0.429 / 2% | 0.340 / 2% |

Zero-shot swings by 11 points and malformed rate by 3-49% depending on wording (the short prompt
without an output example makes the model drop the quotes). Five-shot moves at most 1.6 points -
the demonstrations decide the format and the label prior, the wording barely matters.

### Threshold decoding applied to prompting (dev)

5-shot + t=0.15: EN 0.415 (greedy 0.415), RO 0.337 (greedy 0.340). No gain. The prompted model
already outputs 1.2-1.5 labels per line, so it isn't collapsed the way the fine-tuned one is, and its
per-label probabilities aren't calibrated for this task. So the PEFT advantage survives a
decoding-matched comparison: it isn't a decoding artefact.

### Baselines (test, micro-F1)

zero 0.000, majority (always anger) 0.230, random at train label frequency 0.197.

### Error analysis (test RO, bilingual LoRA t=0.15 vs 5-shot)

485 rows: exact match LoRA-only 34, prompt-only 37, both wrong 398. Exact match is similar, the F1
gap comes from partial credit on multi-label rows. Patterns:
- LoRA over-predicts under the threshold ("Mi-am facut bagajele" gold joy -> anticipation,joy,trust).
  That is the recall/precision trade the threshold buys.
- Prompting emits non-permitted names ("anguish"), dropped by the parser as the brief requires.
- Context-free lines are unlabelable: "Credeam ca sunt cel mai bun jucator din lume" (I thought I was
  the best player in the world) gold = sadness. Only makes sense with the previous line.
- Projection artefacts: "Am spart-o" gold anger,fear comes from English "Well, that broke that up".
- Test contains near-duplicates differing only in diacritics ("Cooper, da-mi cuiele aici!" twice);
  our dedup is on English text so both land in the same partition, which is what matters.

### What the numbers say

1. Decoding is the biggest single effect: +9 to +11 points for the same adapter. Bigger than any
   choice of training language.
2. PEFT beats prompting once decoding is fixed: +11 EN, +8 RO over best five-shot.
3. Five demos are worth ~+12 points over zero-shot, mostly by fixing the output format and the label
   prior (zero-shot over-predicts anticipation, never says joy/sadness/surprise).
4. Zero-shot on Romanian = majority baseline. Not useful.
5. Cross-lingual gap EN -> RO: ~7 points for Qwen prompting, ~10 for Qwen LoRA, ~0 for EuroLLM.
   EuroLLM is worse overall but has no gap - pretraining coverage, not culture (brief warns against
   the culture claim).
6. Bilingual > RO-only > EN-only on Romanian (0.399 / 0.371 / 0.375). Adding English to Romanian
   training helps Romanian; Romanian-only costs 5 points on English.
7. Surprise is the hardest label everywhere. Disgust second. Anger/anticipation easiest (most frequent).
8. The 110M XLM-R encoder is within 1 point of the 1.5B decoder on Romanian at 1/50th the cost.

## Status (2026-09-21)

Done: split, 5 required conditions on dev and test, 3 seeds, second model, threshold sweep,
baselines, cleanup, README.

All experiments done. Not done:
- report (8-12 pages, structure in brief section 6)
- presentation
- figures: label distribution, threshold sweep curve, per-label F1 heatmap

Optional extensions if there's time: Qwen2.5-7B QLoRA, Danish-projected lines as extra RO training
data, Romanian-language instructions.

## Where results live

`runs/` is gitignored. The full set is on Habrok at `/scratch/s5560535/LLM-2026-Group6/runs/`.
One folder per run: `preds.jsonl` (every row: text, raw output, parsed pred, gold, probs) and
`metrics.json` (all metrics + config), or `train_config.json` + `best/` adapter for training runs.
`summary.py` prints everything as one table.

## Environment gotchas

- torch is pinned to the cu126 wheel index in `pyproject.toml` because the laptop driver (550) only
  supports CUDA 12.x. Works on newer drivers too.
- WSL needs `python3.12-dev` for triton to compile its helper.
- On Habrok: venv and HF cache go on `/scratch`, not home (quota). `run_all.sh` sets that up.
  The VS Code server on the Habrok portal already runs inside a GPU job, so `bash run_all.sh` works
  directly; `sbatch` only for jobs longer than the session.
