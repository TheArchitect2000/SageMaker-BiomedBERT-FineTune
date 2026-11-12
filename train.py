import argparse
import json
import os
from pathlib import Path

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

LABELS = [
    "O", "B-DRUG", "I-DRUG", "B-EXCIPIENT", "I-EXCIPIENT",
    "B-DOSAGE_FORM", "I-DOSAGE_FORM", "B-STRENGTH", "I-STRENGTH",
    "B-ROUTE", "I-ROUTE", "B-PROCESS", "I-PROCESS",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    files = {
        "train": str(Path(os.environ["SM_CHANNEL_TRAIN"]) / "train.jsonl"),
        "validation": str(Path(os.environ["SM_CHANNEL_VALIDATION"]) / "validation.jsonl"),
    }
    ds = load_dataset("json", data_files=files)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    label2id = {label: i for i, label in enumerate(LABELS)}
    id2label = {i: label for label, i in label2id.items()}

    def tokenize(batch):
        encoded = tokenizer(batch["tokens"], is_split_into_words=True,
                            truncation=True, max_length=args.max_length)
        aligned = []
        for row, tags in enumerate(batch["ner_tags"]):
            word_ids = encoded.word_ids(batch_index=row)
            labels, previous = [], None
            for w in word_ids:
                # Label only the first sub-token of each word; ignore the rest (-100).
                labels.append(-100 if w is None or w == previous else label2id[tags[w]])
                previous = w
            aligned.append(labels)
        encoded["labels"] = aligned
        return encoded

    tokenized = ds.map(tokenize, batched=True, remove_columns=ds["train"].column_names)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_id, num_labels=len(LABELS), id2label=id2label, label2id=label2id
    )
    seqeval = evaluate.load("seqeval")

    def metrics(result):
        predictions = np.argmax(result.predictions, axis=2)
        predicted, references = [], []
        for pred_row, label_row in zip(predictions, result.label_ids):
            keep = [(id2label[int(p)], id2label[int(y)]) for p, y in zip(pred_row, label_row) if y != -100]
            predicted.append([p for p, _ in keep])
            references.append([y for _, y in keep])
        scores = seqeval.compute(predictions=predicted, references=references)
        return {"precision": scores["overall_precision"], "recall": scores["overall_recall"],
                "f1": scores["overall_f1"], "accuracy": scores["overall_accuracy"]}

    training_args = TrainingArguments(
        output_dir="/opt/ml/checkpoints", evaluation_strategy="epoch", save_strategy="epoch",
        learning_rate=args.learning_rate, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size, num_train_epochs=args.epochs,
        weight_decay=0.01, warmup_ratio=0.1, fp16=True, load_best_model_at_end=True,
        metric_for_best_model="f1", greater_is_better=True, logging_steps=5,
        report_to="none", seed=args.seed, save_total_limit=2,
    )
    trainer = Trainer(
        model=model, args=training_args, tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        train_dataset=tokenized["train"], eval_dataset=tokenized["validation"],
        compute_metrics=metrics,
    )
    trainer.train()
    results = trainer.evaluate()
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    (model_dir / "metrics.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

