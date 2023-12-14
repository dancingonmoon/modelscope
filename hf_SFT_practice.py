from datasets import load_dataset
import transformers
import numpy as np
import torchmetrics

# import os
# print(os.cpu_count())

dataset_dir = '../datasets/rotten_tomatoes'
dataset = load_dataset(path=dataset_dir,)
# print(dataset)

def add_prefix(example):
    example["text"] = 'My sentence: ' + example["text"]
    return example
def add_prefix_batch(batch):
    texts = []
    for i, text in enumerate(batch['text']):
        text = 'My sentence No.' + str(i) + ' :' + text
        texts.append(text)
    batch_ = dict(text=texts, label=batch['label'],)
    return batch_

def compute_metrics(p: transformers.EvalPrediction) -> dict:
    preds,labels=p
    preds = np.argmax(preds, axis=-1)
    #print('shape:', preds.shape, '\n')
    # precision, recall, f1, _ = precision_recall_fscore_support(lables.flatten(), preds.flatten(), average='weighted', zero_division=0)
    precision = torchmetrics.functional.precision(preds,labels)
    recall = torchmetrics.functional.recall(preds,labels)
    f1 = torchmetrics.functional.f1_score(preds,labels)
    accuracy = torchmetrics.functional.accuracy(preds,labels)
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

if __name__ == '__main__':
    # map奇怪的参数: batched=True 打开时,并不是map后的dataset有了batch,而只是为了并行处理,提高map_fn的运行效率;map后生成的
    # dataset并没有batch,shape没有变化;
    dataset_processed = dataset.map(add_prefix_batch,batched=True,batch_size=3,num_proc=8)
    print(next(iter(dataset['test'])))
    print(next(iter(dataset_processed['test'])))

