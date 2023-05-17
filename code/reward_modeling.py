from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Sequence
import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy
from torch.utils.data import Dataset
import transformers
import copy
from tqdm.auto import tqdm
from peft import LoraConfig, TaskType, get_peft_model

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
max_length = 720
padding_strategy = "right"
IGNORE_INDEX = -100
output_dir = "reward_model_weights"


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        data_dict = preprocess(raw_dataset, tokenizer)
        self.input_ids_j = data_dict["input_ids_j"]
        self.input_ids_k = data_dict["input_ids_k"]
        self.attention_mask_j = data_dict["attention_mask_j"]
        self.attention_mask_k = data_dict["attention_mask_k"]

    def __len__(self):
        return len(self.input_ids_j)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids_j=self.input_ids_j[i],
            attention_mask_j=self.attention_mask_j[i],
            input_ids_k=self.input_ids_k[i],
            attention_mask_k=self.attention_mask_k[i],
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        input_embeddings[-num_new_tokens:] = input_embeddings_avg


# Turn the dataset into pairs of post + summaries, where text_j is the preferred question + answer and text_k is the other.
# Then tokenize the dataset.
def preprocess(examples, tokenizer):
    new_examples = {
        "input_ids_j": [],
        "attention_mask_j": [],
        "input_ids_k": [],
        "attention_mask_k": [],
    }
    # with tqdm(total=len(examples["response_j"])) as pbar:
    final_dict = _tokenize_fn(examples["response_j"], examples["response_k"], tokenizer)

    new_examples["input_ids_j"] = final_dict["input_ids_j"]
    new_examples["attention_mask_j"] = final_dict["attention_mask_j"]
    new_examples["input_ids_k"] = final_dict["input_ids_k"]
    new_examples["attention_mask_k"] = final_dict["attention_mask_k"]
    # pbar.update(1)
    return new_examples


# We need to define a special data collator that batches the data in our j vs k format.
def _tokenize_fn(
    strings_j: Sequence[str],
    strings_k: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list_j = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )
        for text in strings_j
    ]
    print("Batch J responses are tokenized")
    tokenized_list_k = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )
        for text in strings_k
    ]
    print("Bacth K responses are tokenized!")
    input_ids_j = [tokenized.input_ids[0] for tokenized in tokenized_list_j]
    input_ids_k = [tokenized.input_ids[0] for tokenized in tokenized_list_k]
    attention_mask_j = [tokenized.attention_mask[0] for tokenized in tokenized_list_j]
    attention_mask_k = [tokenized.attention_mask[0] for tokenized in tokenized_list_k]
    print("All tokenizations are done")
    return dict(
        input_ids_j=input_ids_j,
        input_ids_k=input_ids_k,
        attention_mask_j=attention_mask_j,
        attention_mask_k=attention_mask_k,
        return_loss=True,
    )


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # print("check instance")
        # print(instances)
        input_ids_j = [instance["input_ids_j"] for instance in instances]
        input_ids_j = torch.nn.utils.rnn.pad_sequence(
            input_ids_j, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        input_ids_k = [instance["input_ids_k"] for instance in instances]
        input_ids_k = torch.nn.utils.rnn.pad_sequence(
            input_ids_k, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        return dict(
            input_ids_j=input_ids_j,
            attention_mask_j=input_ids_j.ne(self.tokenizer.pad_token_id),
            input_ids_k=input_ids_k,
            attention_mask_k=input_ids_k.ne(self.tokenizer.pad_token_id),
            return_loss=True,
        )


class RewardTrainer(Trainer):
    # Define how to compute the reward loss. We use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards_j = model(
            input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"]
        )[0]
        rewards_k = model(
            input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"]
        )[0]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss


# Load the dataset using the HuggingFace dataset library
extensions = "json"
KEY_INSTANCES = "instances"
raw_dataset = load_dataset(
    extensions,
    data_files=["final_pairwise_data.json"],
    field=KEY_INSTANCES,
    split="train",
    use_auth_token=None,
)

print(len(raw_dataset["response_j"]))
print(len(raw_dataset["response_k"]))

# Load the value-head model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained(
    "decapoda-research/llama-7b-hf",
    model_max_length=max_length,
    padding_side=padding_strategy,
    use_fast=False,
)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = AutoModelForSequenceClassification.from_pretrained(
    "decapoda-research/llama-7b-hf", num_labels=1, torch_dtype=torch.bfloat16
)
# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()

print("Loaded in model and tokenizers")
if tokenizer.pad_token is None:
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        tokenizer=tokenizer,
        model=model,
    )

tokenizer.add_special_tokens(
    {
        "eos_token": DEFAULT_EOS_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "unk_token": DEFAULT_UNK_TOKEN,
    }
)

ds_config = "config/ds_config_zero3.json"
data_module = make_supervised_data_module(tokenizer=tokenizer)
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="no",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    learning_rate=2e-5,
    weight_decay=0,
    num_train_epochs=3,
    warmup_ratio=0,
    logging_strategy="steps",
    logging_first_step=True,
    save_strategy="epoch",
    save_total_limit=3,
    seed=42,
    run_name="wandb",
    load_best_model_at_end=False,
    greater_is_better=False,
    deepspeed=ds_config,
    log_on_each_node=False,
    logging_steps=1,
    fp16=True,
    lr_scheduler_type="cosine",
)

# Train the model
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=data_module["train_dataset"],
    eval_dataset=None,
    tokenizer=tokenizer,
    data_collator=data_module["data_collator"],
    preprocess_logits_for_metrics=None,
)

trainer.train()
print("Saving last checkpoint of the model")
model.save_pretrained(output_dir)
