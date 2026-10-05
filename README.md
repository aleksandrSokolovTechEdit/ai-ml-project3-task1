# CareScribe: fine-tuning a clinical summary LLM

Fine-tune `Qwen/Qwen2.5-1.5B-Instruct` with QLoRA to turn de-identified clinical case notes into plain-language after-visit summaries.

- `finetune_spec.md` — the fine-tuning spec (Task 1)
- `DATASET_CARD.md` — dataset card (Task 2)
- `REPORT.md` — delivery report, filled in as the project goes

## Layout

```
src/        curation, PHI scan, training, evaluation scripts
configs/    one YAML per training configuration
data/       raw/ corpus; the built dataset is DVC-tracked
eval/       grading sheets, unmask keys, metric outputs
notebooks/  exploration, colab_train.ipynb
assets/     images referenced from the report
```

## Reproduce

```bash
pip install -r requirements.txt
python src/check_env.py
```
