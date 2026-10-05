# Fine-Tuning Spec: CareScribe after-visit summaries

## 1. Requirement
Input: a clinical case note, free text, published and de-identified.
Output: an after-visit summary in CareScribe's five-section format, written in a plain-language register a patient can read without a clinician in the room.
Hard constraint: every fact in the summary (diagnosis, medication, dose, duration, follow-up date) is traceable to the input note; the model never adds a fact the note does not contain.

## 2. Base model
- **Model id:** `Qwen/Qwen2.5-1.5B-Instruct` (1.54B parameters, 1.31B non-embedding; 28 layers, hidden size 1536, GQA with 12 query / 2 key-value heads; context 32k).
- **Chat template turn markers** (pasted from `tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)`):

```
<|im_start|>system
You are CareScribe, a clinical documentation assistant.<|im_end|>
<|im_start|>user
{instruction}

{note}<|im_end|>
<|im_start|>assistant
```

  The assistant turn ends with `<|im_end|>`, which is also the EOS token used at generation. Training data is formatted with this exact template, and the loss is masked to the assistant turn only.
- **Why this model and this size:** it is instruct-tuned (already follows instructions and ships with a chat template), it holds a multi-section format after a short fine-tune, it downloads without a login (Apache 2.0), and under QLoRA it fits in the 8 GB GPU I train on (section 6) with headroom. A 7B model would write better prose but needs a 16 GB card, which I don't have. `Qwen/Qwen2.5-0.5B-Instruct` is the fallback if memory or time runs out.

## 3. Training plan
| Parameter | Value |
|---|---|
| Method | QLoRA |
| Base quantization | 4-bit NF4, double quantization, bf16 compute dtype |
| LoRA rank / alpha / dropout | r = 16, alpha = 32, dropout = 0.05 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (names confirmed with `model.named_modules()`) |
| Trainable parameters | ≈ 18.5M (≈ 1.2 % of the model) |
| Max sequence length | 1024 tokens (prompt + summary; see section 4) |
| Epochs | 2 |
| Micro-batch / grad accumulation | 4 × 4 → **effective batch size 16** |
| Optimizer / LR | paged AdamW 8-bit, lr 2e-4, cosine schedule, 3 % warmup |
| Precision | bf16 (compute capability 8.9 supports it) |
| Gradient checkpointing | on |
| Checkpointing | every 50 steps, keep last 2 (for the interruption-and-resume drill) |

**Why not full fine-tuning:** for 1.5B parameters, weights (3 GB) + gradients (3 GB) + Adam momentum and variance in fp32 (6 GB + 6 GB) = 18 GB before activations, more than twice what my GPU has.

**Expected peak memory:** ≈ 1.2 GB for the 4-bit base (incl. bf16 embeddings) + ≈ 0.2 GB for adapters, their gradients and 8-bit optimizer states + ≈ 2.5–3.5 GB of activations at micro-batch 4 × 1024 tokens with gradient checkpointing → **≈ 5 GB expected, hard ceiling 7 GB**. Confirmed with `torch.cuda.max_memory_allocated()` on the smoke test in Task 3.

**Output artifact:** a PEFT LoRA adapter (`adapter_model.safetensors` + `adapter_config.json`, ≈ 70 MB), not a merged model. Logged to MLflow together with the dataset hash.

## 4. Data requirements
- **Pair count:** ≈ 1,700 pairs in `data/raw/`; expected ≈ 1,500 after the gates.
- **Fixed instruction** (identical in every example and in the deployed app):
  > Write an after-visit summary of the clinical note below for the patient. Use plain language and the five CareScribe sections. Include only information stated in the note.
- **Summary structure** (five sections, in this order, in every example):
  1. Why you came in
  2. What we found
  3. Your treatment and medicines
  4. What to do next
  5. When to get help
- **Curation gates:**
  - Exact duplicates removed (hash of normalized note text), near-duplicates removed at MinHash Jaccard ≥ 0.9 on 5-gram shingles.
  - Minimum summary length: 60 tokens.
  - Maximum combined length (prompt + note + summary): 1024 tokens, set from the 99th percentile of a 300-pair sample tokenized with the Qwen2.5 tokenizer (p99 ≈ 960).
  - Summary-to-note ratio: drop pairs where the summary has more tokens than the note.
  - Structure check: drop pairs whose summary doesn't contain all five section headers in order.
- **PHI gate:** `src/phi_scan.py` scans the dataset and the whole repository with regexes for names after titles (Mr/Mrs/Dr), dates of birth, phone numbers, emails, street addresses, MRN/ID-like numbers and ZIP codes. It must report zero hits; I keep a test that plants one fake violation and confirms the scan fails (exit code 1) before trusting a pass.
- **Splits** (after deduplication, seed 42): train 80 % (≈ 1,200), validation 10 % (≈ 150), held-out 10 % (≈ 150). The held-out split is written once, DVC-tracked, and opened exactly once at final evaluation.

## 5. Success criteria
| Instrument | What it answers | What it cannot see | Bar |
|---|---|---|---|
| ROUGE-L | Did the model learn the format and phrasing? | Truth; valid paraphrases | Tuned > base on held-out |
| BERTScore (F1, `roberta-large`, hashcode logged) | Does it say the same thing as the reference? | Fluent unsupported facts | Tuned > base on held-out |
| Hallucination rate (doses, durations, medication names extracted and checked against the note; unsupported / total, averaged per summary) | Is every checkable claim supported by the note? | Negation, unit conversion, style, usefulness | Tuned ≤ base |
| Clinician rubric, graded blind | Would a clinic accept this draft? | Doesn't scale; depends on protocol | Blind A/B on 25 held-out notes, shuffled with a logged seed; tuned preferred on format and plain language with no loss on accuracy |

**Evaluation protocol:** same held-out inputs, same prompt and chat template, same decoding parameters for both models.
**Decoding parameters (fixed):** greedy decoding (`do_sample=False`), `max_new_tokens=512`, `repetition_penalty=1.0`, `seed=42`.
**Reproducibility:** every number in the report comes from `src/evaluate.py` and a logged MLflow run.
**Honest-miss clause:** if a bar is missed, the report documents why with evidence and the debugging steps tried; no number is reported without the run behind it.

## 6. Compute budget
- **Machine:** laptop with NVIDIA GeForce RTX 4060 Laptop GPU; `nvidia-smi` reports 8188 MiB; compute capability 8.9; CUDA 12.1; tier reported by `src/check_env.py`: GPU ≥ 6 GB (reference plan).
- **Arithmetic per run:** 1,200 pairs × 2 epochs / 16 effective batch = 150 optimizer steps. Planning assumption: ≈ 2.5 s per optimizer step (4 micro-batches) → ≈ 6–7 min of training. Validation generation: 150 notes × ≈ 6 s ≈ 15 min. **≈ 0.4 GPU h per run.**
- **Runs expected:** 1 smoke test + 4 training runs (rank / LR / epochs variations) + 1 final held-out evaluation ≈ 6 runs × 0.4 h ≈ 2.4 h.
- **Cap: 5 GPU hours total**, set before the first run (≈ 2× the estimate to cover a restarted run and debugging). Fallback if the GPU is unavailable: `notebooks/colab_train.ipynb` on a T4, same scripts and config.

## 7. Out of scope
- Full-parameter fine-tuning (memory, see section 3) and models above 2B parameters.
- Merging the adapter into the base weights, quantizing for deployment, or serving the model.
- Notes in languages other than English, and specialties not represented in the corpus.
- Clinical decision support: the model summarizes the note; it does not diagnose, recommend treatment or correct the note.
- Real patient data: the project uses only the published, de-identified corpus.
- RLHF / DPO or any preference tuning.

## 8. Open questions
- Is the reference summary style consistent across sources in the corpus, or will curation need a style filter? (answered in Task 2)
- Does the hallucination extractor handle doses written as words ("twice a day") as well as numbers?
- Who grades the rubric blind, and is one grader enough or do we need two with agreement measured?
- Is 1024 tokens enough for the longest notes, or will truncation drop facts the summary depends on?

## 9. Change log
The change log opens in Task 2.
