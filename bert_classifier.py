import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam, BertConfig
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import f1_score
from enum import Enum
import csv
import random

CACHE = "./models"
BERT_MODEL = "bert-base-cased"
MAX_SEQ_LENGTH = 180


class Task(Enum):
    A = "A"
    B = "B"
    C = "C"


TASK_LABELS = {
    Task.A: {
        "NOT": 0,
        "OFF": 1
    },
    Task.B: {
        "TIN": 0,
        "UNT": 1
    },
    Task.C: {
        "IND": 0,
        "GRP": 1,
        "OTH": 2
    }
}
TRAIN = "./data/offenseval-training-v1.tsv"
TEST = {
    Task.A: "./data/test_set_taska.tsv",
    Task.B: "./data/test_set_taskb.tsv",
    Task.C: "./data/test_set_taskc.tsv",
}


def load_test_dataset(task):
    ids = []
    x = []
    with open(TEST[task], 'rt') as test:
        for inp in csv.reader(test, delimiter='\t'):
            x.append(inp[1])
            ids.append(inp[0])
        return ids, x


def load_train_dataset(task):
    with open(TRAIN, 'rt') as entire_set:
        tsvin = csv.reader(entire_set, delimiter='\t')

        x = []
        y = []
        for i, inp in enumerate(tsvin):
            if task == Task.A:
                x.append(inp[1])
                y.append(TASK_LABELS[task][inp[2]])
            elif task == Task.B and inp[3] != "NULL":
                x.append(inp[1])
                y.append(TASK_LABELS[task][inp[3]])
            elif task == Task.C and inp[4] != "NULL":
                x.append(inp[1])
                y.append(TASK_LABELS[task][inp[4]])

        return np.array(x), np.array(y)


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def convert_examples_to_features(x, y, max_seq_length, tokenizer):
    features = []
    for i, x in enumerate(x):
        tokens = tokenizer.tokenize(x)
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        label_id = y[i]

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id))
    return features


class ClassificationModel:
    def __init__(self, task, val=0.1, bert_model=BERT_MODEL, gpu=False, seed=0):
        self.gpu = gpu
        self.task = task
        self.bert_model = bert_model
        self.x_train, self.y_train = load_train_dataset(self.task)
        self.x_val = np.random.choice(self.x_train, size=(int(val * len(self.x_train)),), replace=False)
        self.y_val = np.random.choice(self.y_train, size=(int(val * len(self.x_train)),), replace=False)
        self.x_test_ids, self.x_test = load_test_dataset(self.task)
        self.num_classes = len(TASK_LABELS[task])

        self.model = None
        self.optimizer = None
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model)

        self.plt_x = []
        self.plt_y = []

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.gpu:
            torch.cuda.manual_seed_all(seed)

    def __init_model(self):
        if self.gpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        print(torch.cuda.memory_allocated(self.device))

    def new_model(self):
        self.model = BertForSequenceClassification.from_pretrained(self.bert_model, num_labels=self.num_classes)
        self.__init_model()

    def load_model(self, path_model, path_config):
        self.model = BertForSequenceClassification(BertConfig(path_config), num_labels=self.num_classes)
        self.model.load_state_dict(torch.load(path_model))
        self.__init_model()

    def save_model(self, path_model, path_config):
        torch.save(self.model.state_dict(), path_model)
        with open(path_config, 'w') as f:
            f.write(self.model.config.to_json_string())

    # noinspection PyArgumentList
    def train(self, epochs, plot_path, batch_size=32, lr=5e-5, model_path=None, config_path=None):
        model_params = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = BertAdam(optimizer_grouped_parameters, lr=lr, warmup=0.1,
                                  t_total=int(len(self.x_train) / batch_size) * epochs)

        nb_tr_steps = 0
        train_features = convert_examples_to_features(self.x_train, self.y_train, MAX_SEQ_LENGTH, self.tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        _, counts = np.unique(self.y_train, return_counts=True)
        class_weights = [sum(counts) / c for c in counts]
        example_weights = [class_weights[e] for e in self.y_train]
        sampler = WeightedRandomSampler(example_weights, len(self.y_train))
        train_dataloader = DataLoader(train_data, sampler=sampler, batch_size=batch_size)

        self.model.train()
        for e in range(epochs):
            print(f"Epoch {e}")
            f1, acc = self.val()
            print(f"\nF1 score: {f1}, Accuracy: {acc}")
            if model_path is not None and config_path is not None:
                self.save_model(model_path, config_path)
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                loss = self.model(input_ids, segment_ids, input_mask, label_ids)
                loss.backward()

                self.plt_y.append(loss.item())
                self.plt_x.append(nb_tr_steps)
                self.save_plot(plot_path)

                nb_tr_steps += 1
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.gpu:
                    torch.cuda.empty_cache()

    def val(self, batch_size=32, test=False):
        eval_features = convert_examples_to_features(self.x_val, self.y_val, MAX_SEQ_LENGTH, self.tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

        f1, acc = 0, 0
        nb_eval_examples = 0

        for input_ids, input_mask, segment_ids, gnd_labels in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)

            predicted_labels = np.argmax(logits.detach().cpu().numpy(), axis=1)
            acc += np.sum(predicted_labels == gnd_labels.numpy())
            tmp_eval_f1 = f1_score(predicted_labels, gnd_labels, average='macro')
            f1 += tmp_eval_f1 * input_ids.size(0)
            nb_eval_examples += input_ids.size(0)

        return f1 / nb_eval_examples, acc / nb_eval_examples

    def save_plot(self, path):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(self.plt_x, self.plt_y)

        ax.set(xlabel='Training steps', ylabel='Loss')

        fig.savefig(path)
        plt.close()

    def create_test_predictions(self, path):
        eval_features = convert_examples_to_features(self.x_test, [-1] * len(self.x_test), MAX_SEQ_LENGTH,
                                                     self.tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=16)

        predictions = []
        inverse_labels = {v: k for k, v in TASK_LABELS[self.task].items()}

        for input_ids, input_mask, segment_ids, gnd_labels in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)

            predictions += [inverse_labels[p] for p in list(np.argmax(logits.detach().cpu().numpy(), axis=1))]
        with open(path, "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for i, prediction in enumerate(predictions):
                writer.writerow([int(self.x_test_ids[i]), prediction])

        return predictions


if __name__ == "__main__":
    PATH_CONFIG = "./results/b-uncased-4-epochs/config"
    PATH_STATE = "./results/b-uncased-4-epochs/state"
    PLOT_PATH = "./plot.png"

    cm = ClassificationModel(Task.C, gpu=False, seed=0, val=0.05)
    cm.load_model(PATH_STATE, PATH_CONFIG)
    cm.create_test_predictions("./c_pred.csv")
