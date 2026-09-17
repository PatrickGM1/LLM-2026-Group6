Cross-Lingual Multi-Label Emotion Classification Using
Large Language Models
COURSE PROJECT BRIEF
Course Introduction to Large Language Models (LLMs): Course Project
Groups Three students per group
Duration Weeks 1–8 of the course
Deliverables Report, codebase, and presentation
1. Objective & Motivation
Large Language Models are strikingly capable at language understanding and generation,
but how well that extends to multi-label tasks across many languages and to domain-
specific data like emotion in dialogue is still an open research question. Emotion is also
expressed differently from one language and culture to the next, which makes cross-lingual
emotion detection a demanding test of what these models can really do.
This project investigates how well LLMs classify emotions across languages using the XED
multilingual emotion dataset. You will compare two adaptation approaches, prompt
engineering and parameter-efficient fine-tuning (PEFT) and examine their cross-lingual
transfer to an assigned language. You will assess where the models succeed and struggle,
including prompt sensitivity, computational cost, differences in pretrained language
coverage, and limitations introduced by projected annotations.
“Multi-label” means that a single subtitle line can express several emotions at once—for
example, both surprise and fear—rather than exactly one emotion. This makes the task more
complex than single-label classification and requires the multi-label evaluation metrics
defined in §7.
2. Dataset
The XED dataset (Helsinki-NLP/XED on GitHub) provides emotion-annotated movie
subtitles.

Multilingual: English and Finnish contain human annotations; most additional-language
data were created by projecting labels to aligned subtitle lines. The projected labels
should therefore be treated as projected reference labels rather than independent
human judgements in each language.
Multi-label: Each subtitle line can be associated with multiple emotions from Plutchik's
eight core categories: anger, anticipation, disgust, fear, joy, sadness, surprise, and trust.
Sentence-level subtitles: Each example is a subtitle line. Surrounding dialogue context is
not part of the standard input and may only be introduced as a clearly documented
optional extension.
The available projected-language files vary enormously in size. At the low end, Macedonian
has 300 labelled lines, with Vietnamese (956), Slovak (975), and Icelandic (977) close
behind. At the high end, Brazilian Portuguese has 12,295 lines and Spanish 11,303—more
than a 40-fold difference. The language allocation in section 3 is designed to expose the
cohort to this variation.
3. Methodology & Tasks
You will work with open-weight language models that can be run through tools such as
Hugging Face Transformers. Select models with suitable multilingual coverage and a size
that can be prompted and fine-tuned using the available course infrastructure.
Language Allocation
Each group works on English plus one assigned XED language. Across the cohort,
assignments should cover a range of dataset sizes from a few hundred labelled lines to more
than twelve thousand. The cohort may then explore whether the observed gap between
fine-tuning and prompting is associated with the amount of target-language training data.
The languages below are grouped into five resource bands. The examples are illustrative rather
than exhaustive.
Approx.
Band Example languages
annotated lines
1 — extreme Macedonian (300), Vietnamese (956), Slovak (975),
under ~1,000
low Icelandic (977)
Danish, Russian, Bosnian, Slovenian, Arabic,
2 — very low ~1,000–4,400
Norwegian
Hebrew, Swedish, Dutch, German, Hungarian,
3 — mid-low ~4,400–6,000
Croatian

4 — mid-
~6,000–7,300 Czech, Italian, Bulgarian, Polish, Portuguese, French
high
~8,000 and Greek, Finnish, Serbian, Turkish, Romanian, Spanish,
5 — high
above Brazilian Portuguese
How languages are assigned
Groups are distributed across the five bands so that the cohort covers the broadest
feasible range.
At the start of Week 1, a short survey collects which of these languages, if any,
someone in your group can already read.
Where a group has a reader for a language in its band, that preference is taken into
account; the instructor confirms the final assignments and settles any overlaps between
groups.
If no one in your group reads the assigned language, that is not a problem for
quantitative evaluation. The assigned-language outputs are evaluated against XED’s
projected reference labels. You may inspect aligned English lines to help interpret
examples, but those lines do not independently validate translation quality, cultural
appropriateness, or the emotional meaning of the assigned-language subtitle.
Required Data-Splitting Protocol
Create one fixed 70/20/10 train/development/test split using a documented seed and a
multi-label stratification procedure.
Split by XED alignment identifier before constructing the English and assigned-language
datasets. Parallel versions of the same subtitle must remain in the same partition.
For the required experimental matrix, use matched English–assigned-language pairs: the
English dataset should contain the English lines aligned with the assigned-language
examples.
Detect exact duplicate lines and ensure that duplicates do not occur across partitions.
Use the development set for prompt selection, hyperparameter selection, checkpoint
selection, and all other design decisions. Use the test set only for the final reported
evaluation.
Draw all few-shot demonstrations from the training set only.
Minimum Required Experimental Matrix
Use one primary base model for all prompting and PEFT conditions so that the comparisons
remain interpretable. Use a second model only for the zero-shot and few-shot prompting
conditions. The following experiments are mandatory:

| Condition | Model(s) | Training / Demonstrations | Evaluate on |
| --------- | -------- | ------------------------- | ----------- |
English +
| Zero-shot | Primary + |      |          |
| --------- | --------- | ---- | -------- |
|           |           | None | assigned |
| prompting | second    |      |          |
language
English +
| Five-shot | Primary + | Five fixed, aligned training examples |     |
| --------- | --------- | ------------------------------------- | --- |
assigned
| prompting | second | in the evaluation language |     |
| --------- | ------ | -------------------------- | --- |
language
English +
English-only
|     | Primary | English training set | assigned |
| --- | ------- | -------------------- | -------- |
PEFT
language
| Target-       |         |                                | Assigned |
| ------------- | ------- | ------------------------------ | -------- |
|               | Primary | Assigned-language training set |          |
| language PEFT |         |                                | language |
English +
English + assigned-language
| Bilingual PEFT | Primary |     | assigned |
| -------------- | ------- | --- | -------- |
training sets
language
One PEFT method—LoRA or QLoRA—is required. A second PEFT method, additional
ranks/alpha values, translated instructions, context windows, or controlled downsampling
are optional extensions. Use the development set to select one final PEFT configuration; a
full hyperparameter grid is not required.
Phase 1: LLM Adaptation and Fine-tuning (Weeks 1–5)
Data preparation: Format the English and assigned-language training partitions as
instruction-following or conversational examples. Use exactly the eight permitted
emotion names and preserve the multi-label targets.
Parameter-efficient fine-tuning: Use LoRA or QLoRA to run the three required PEFT
conditions in the experimental matrix: English-only, assigned-language-only, and
bilingual fine-tuning.
Document the adapter targets, rank, alpha, dropout, learning rate, batch size, number
of epochs, maximum sequence length, quantization settings, checkpoint-selection rule,
random seed, and hardware.
Use the development set—not the test set—to select the final configuration.
Evaluation: Evaluate each PEFT condition on the test partitions specified in the matrix
and compare it with the prompting conditions for the same primary base model. The
SVM/BERT values reported by Öhman et al. (2020) may be discussed as historical
reference points, but they must not be treated as directly comparable baselines unless
the data, splits, label space, and metric definition are reproduced.
Qualitative analysis: Analyze concrete test examples where fine-tuning improved or
worsened predictions compared with prompting. Distinguish model errors from possible
subtitle-alignment or label-projection limitations.

Phase 2: Zero-Shot and Few-Shot Prompt Engineering (Weeks 5–6)
Model selection: Use the primary model selected for PEFT and one second model for
prompting only. Justify both choices in terms of model size, multilingual coverage,
licence, and accessibility.
Prompt design: For each model, implement one zero-shot prompt and one five-shot
prompt. Both prompts must:
define the eight permitted emotion labels and explain that one or several labels may be
returned;
require a JSON list containing only the permitted English label names;
use the same English-language instruction template for English and assigned-language
utterances;
use the same five fixed alignment IDs throughout five-shot evaluation, presenting the
English examples for English evaluation and their assigned-language counterparts for
assigned-language evaluation; and
report the complete prompt, decoding parameters, and demonstration-selection
procedure.
Output parsing: Normalize case and whitespace, but do not map arbitrary synonyms to
labels. Remove duplicate permitted labels. Treat malformed outputs or outputs
containing no permitted label as an empty predicted label set; report the malformed-
output rate separately.
Evaluation: Evaluate both prompt conditions on English and the assigned language
using section 7. Compare models, languages, and prompt conditions, and analyse
sensitivity to prompt wording on the development set. Do not redesign the prompt after
inspecting test results.
4. Project Timeline & Syllabus Alignment
How the two project phases line up against the course's weekly lecture content.
Week Lecture Topic Project Activity
Phase 1 kickoff: group formation,
1 Introduction to LLMs language-band assignment (§3), dataset
exploration, data prep begins.
Architectures & Scaling Phase 1 continues: environment setup,
2
Laws baseline model selection.
Phase 1: fine-tuning experiments begin
3 Training LLMs
(SFT concepts now covered).

Phase 1 core work: implement the
Efficiency in LLMs
4 required English-only, target-language,
(LoRA/PEFT/QLoRA)
and bilingual PEFT conditions.
Phase 1 wraps up (evaluation) — Phase 2
5 Prompt Engineering
begins (prompt design).
Knowledge and Phase 2: cross-lingual prompt transfer
6
Reasoning evaluation and analysis.
Multimodal LLMs & Course report, codebase finalization,
7
Future Directions presentation prep, submission.
5. Expected Outcomes / Deliverables
Each group produces three deliverables, all due at the end of Week 7:
Technical report (8–12 pages): a written account of the methods, results, and analysis.
See §6 for the required structure.
Reproducible codebase: well-structured, documented, and runnable code for data
preparation, prompting, PEFT, parsing, and evaluation, hosted on GitHub. It must
include a README, environment or dependency file, configuration files, fixed seeds,
and commands for reproducing the principal results. Do not commit model weights,
access tokens, or unnecessary copies of the dataset.
Project presentation: a concise presentation summarising the methodology, key
findings, limitations, and conclusions.
6. Report Structure
The technical report should be 8–12 pages, excluding references and appendices, and cover
the elements below. Treat this as a scaffold rather than a fixed form: you may merge or
reorder sections where this makes your argument clearer, as long as a reader can find each
element. The page figures are rough guidance for the 8–12-page budget, not hard limits.
Section ~Pages What it covers
The problem, why multilingual and multi-label emotion
1. Introduction &
~1 detection is hard, your objectives, and your group’s
Motivation
assigned language and band.
A short account of XED and the approaches you use
2. Background ~1 (PEFT, prompting). Keep it brief — depth belongs in
later sections.

The XED data for your language, projected-label
3. Data & Task
~1 status, label space, leakage controls,
Setup
train/development/test split, and data formatting.
The models and selection rationale; PEFT setup and
4. Methodology ~2 training conditions; prompt templates and
demonstrations; decoding settings; and output parser.
The metrics in section 7, random seeds, evaluation
5. Evaluation script, malformed-output policy, and controlled
~0.5–1
Protocol comparisons. Published XED values may be included
separately as historical reference points.
Your quantitative findings in clear tables and charts,
6. Results ~2
using the multi-label metrics.
Error analysis with concrete examples; cross-lingual
transfer, prompt sensitivity, and the PEFT-versus-
7. Analysis & prompting trade-off; malformed outputs; projected-
~2
Discussion label limitations; and other biases or limitations. Do
not attribute cross-language differences to culture
without direct evidence.
8. Conclusion & What you conclude, and what you would do with more
~0.5
Future Work time.
9. References (+ Full references (any length). An appendix may hold the
—
optional Appendix) prompts you used, extra tables, and similar material.
7. Evaluation Metrics
Primary: micro-averaged F1 and sample-averaged Jaccard index.
Secondary: macro-averaged F1, micro-averaged precision and recall, and Hamming loss.
Remember that lower Hamming loss is better.
Diagnostic: per-emotion precision, recall, F1, and support; exact-match ratio; and
malformed-output rate for prompted models.
Compute every metric from the same binarized eight-label predictions. Use a fixed label
order and report the software implementation and version. When a metric is undefined
because a class has no positive examples or predictions, use a documented zero-division
policy consistently. Report the random seed for every run; if conditions are repeated, report
the mean and standard deviation.
Absolute performance is not graded directly. Assessment should reward methodological
correctness, reproducibility, interpretation, and critical analysis. See the companion

grading rubric for how these map to report, codebase, and presentation scores.
8. Tools & Technologies (Recommended)
Programming Language: Python
Libraries:
Hugging Face Transformers: for accessing and working with LLMs.
Hugging Face PEFT: for Parameter-Efficient Fine-Tuning (LoRA, QLoRA).
scikit-learn: for evaluation metrics.
pandas and NumPy: for data handling.
Matplotlib and seaborn: for visualisation.
Hugging Face Datasets: for data loading and processing.
