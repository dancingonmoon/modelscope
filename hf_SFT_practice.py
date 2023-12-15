from datasets import load_dataset

import numpy as np
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

model_dir = snapshot_download("AI-ModelScope/bert-base-cased")
dataset_dir = "../datasets/yelp_review_full"
dataset = load_dataset(
        path=dataset_dir,
    )
model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=5)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
def tokenize_function(examples):
    return tokenizer(examples["text"], padding=False, truncation=True)


def add_prefix(example):
    example["text"] = "My sentence: " + example["text"]
    return example


def add_prefix_batch(batch):
    texts = []
    for i, text in enumerate(batch["text"]):
        text = "My sentence No." + str(i) + " :" + text
        texts.append(text)
    batch_ = dict(
        text=texts,
        label=batch["label"],
    )
    return batch_


def compute_metrics(p: transformers.EvalPrediction) -> dict:
    preds, labels = p
    preds = np.argmax(preds, axis=-1)
    # print('shape:', preds.shape, '\n')
    precision = torchmetrics.functional.precision(preds, labels)
    recall = torchmetrics.functional.recall(preds, labels)
    f1 = torchmetrics.functional.f1_score(preds, labels)
    accuracy = torchmetrics.functional.accuracy(preds, labels)
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}


if __name__ == "__main__":
    # map奇怪的参数: batched=True 打开时,并不是map后的dataset有了batch,而只是为了并行处理,提高map_fn的运行效率;map后生成的
    # dataset并没有batch,shape没有变化;
    # dataset_processed = dataset.map(add_prefix_batch,batched=True,batch_size=3,num_proc=8)
    # print(next(iter(dataset['test'])))
    # print(next(iter(dataset_processed['test'])))



    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=8,load_from_cache_file=True)

    # 训练:
    batch_size = 8

    training_args = TrainingArguments(
        output_dir="test_trainer",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        # push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        tokenizer=tokenizer,
        # eval_dataset=tokenized_dataset['test'],
        # compute_metrics=compute_metrics,
    )
    trainer.train()
