from datasets import load_dataset

# import numpy as np
import torch
import torchmetrics
from modelscope import snapshot_download
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import transformers

# import os
# print(os.cpu_count())

# if torch.cuda.is_available():
#     device = torch.device("cuda")  # 使用第一个可用的GPU
# else:
#     device = torch.device("cpu")

model_dir = snapshot_download("AI-ModelScope/bert-base-cased")
dataset_dir = "../datasets/yelp_review_full"
dataset = load_dataset(path=dataset_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=5)
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding=False, truncation=True, max_length=512)


def compute_metrics(p: transformers.EvalPrediction) -> dict:
    preds, labels = p
    preds = torch.from_numpy(preds)
    labels = torch.from_numpy(labels)

    precision = torchmetrics.functional.precision(preds, labels, task='multiclass', num_classes=5, )
    recall = torchmetrics.functional.recall(preds, labels, task='multiclass', num_classes=5, )
    f1 = torchmetrics.functional.f1_score(preds, labels, task='multiclass', num_classes=5, )
    accuracy = torchmetrics.functional.accuracy(preds, labels, task='multiclass', num_classes=5, )
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}


if __name__ == "__main__":
    # map奇怪的参数: batched=True 打开时,并不是map后的dataset有了batch,而只是为了并行处理,提高map_fn的运行效率;map后生成的
    # dataset并没有batch,shape没有变化;
    # dataset_processed = dataset.map(add_prefix_batch,batched=True,batch_size=3,num_proc=8)
    # print(next(iter(dataset['test'])))
    # print(next(iter(dataset_processed['test'])))

    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=8, load_from_cache_file=True)
    train_dataset = tokenized_dataset['train'].shuffle(seed=11).select(range(3000))
    test_dataset = tokenized_dataset['test'].shuffle(seed=11).select(range(1000))
    # 训练:
    batch_size = 8

    training_args = TrainingArguments(
        output_dir="test_trainer",
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        weight_decay=0.01,
        dataloader_num_workers=8,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        group_by_length=True,
        # push_to_hub=True,
    )

    EarlyStopCallback = transformers.EarlyStoppingCallback(early_stopping_patience=20)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStopCallback]
    )
    trainer.train()
