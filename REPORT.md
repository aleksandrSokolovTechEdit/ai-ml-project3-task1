# CareScribe Delivery Report

## 1. Data and PHI discipline
- Provenance and counts: pairs in, duplicates dropped, outliers dropped, pairs out
- The fixed instruction and the summary structure you enforced
- PHI scan: the patterns it checks, the planted-violation test, the passing report
- Split sizes, and the DVC hash of the dataset

## 2. Baseline (validation split)
- Base model id, its chat template, and the decoding parameters you fixed
- Base-model ROUGE-L, BERTScore and hallucination rate on validation
- What the base model already gets right, and where it misses

## 3. Training setup and runs
- LoRA/QLoRA config, sequence length, epochs, effective batch size
- Hardware: GPU model and VRAM (or CPU), precision, and why
- Run history with MLflow run names, dataset hash logged on every run

## 4. Metrics (held-out split, opened once)
- Base against tuned: ROUGE-L, BERTScore, hallucination rate
- The decoding parameters used, identical on both sides
- The BERTScore hashcode, and the extractors the factuality check uses

## 5. Rubric findings
- Protocol: sample size, blinding, shuffle seed
- Per-criterion means, base against tuned
- Two or three side-by-side exhibits
- The out-of-distribution probe: the three notes, and every failure you found
- The style-collapse check

## 6. Compute
- The hours budget you stated before the first run
- Per-run log: hardware, wall-clock hours, peak VRAM
- Total hours against the budget, and the interruption-and-resume drill evidence

## 7. Limitations and retrospective
- What the evaluation cannot see
- What you would do differently
- Any honest miss, with what you tried
- At least one place your AI assistant was wrong, and how you caught it
