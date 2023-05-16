from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
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

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
max_length = 1024
padding_strategy = 'right'

# Load the value-head model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained(
        "decapoda-research/llama-7b-hf",
        model_max_length=max_length,
        padding_side=padding_strategy,
        use_fast=False,
    )
config = AutoConfig.from_pretrained("decapoda-research/llama-7b-hf")
tokenizer.add_special_tokens(
    {
        "eos_token": DEFAULT_EOS_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "unk_token": DEFAULT_UNK_TOKEN,
        "pad_token": DEFAULT_PAD_TOKEN,
    }
)
model = AutoModelForSequenceClassification.from_pretrained(
    "decapoda-research/llama-7b-hf", num_labels=1, torch_dtype=torch.bfloat16
)

# Load the dataset using the HuggingFace dataset library
extensions = "json"
KEY_INSTANCES = "instances"
train_dataset = load_dataset(
    extensions,
    data_files=["final_pairwise_data.json"],
    field=KEY_INSTANCES,
    split="train",
    use_auth_token=None,
)

print(train_dataset)


# Turn the dataset into pairs of post + summaries, where text_j is the preferred question + answer and text_k is the other.
# Then tokenize the dataset.
def preprocess_function(examples):
    new_examples = {
        "input_ids_j": [],
        "attention_mask_j": [],
        "input_ids_k": [],
        "attention_mask_k": [],
    }
    for response_j, response_k in zip(examples["response_j"], examples["response_k"]):
        tokenized_j = tokenizer(response_j, truncation=True)
        tokenized_k = tokenizer(response_k, truncation=True)

        new_examples["input_ids_j"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_j"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_k"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_k"].append(tokenized_k["attention_mask"])

    return new_examples

num_proc=24
original_columns = train_dataset.column_names
# preprocess the dataset and filter out QAs that are longer than script_args.max_length
train_dataset = train_dataset.map(
    preprocess_function, batched=True, num_proc=num_proc, remove_columns=original_columns
)
train_dataset = train_dataset.filter(
    lambda x: len(x["input_ids_j"]) <= max_length and len(x["input_ids_k"]) <= max_length
)

# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_j = []
        features_k = []
        for feature in features:
            features_j.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            features_k.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )
        batch_j = self.tokenizer.pad(
            features_j,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_j": batch_j["input_ids"],
            "attention_mask_j": batch_j["attention_mask"],
            "input_ids_k": batch_k["input_ids"],
            "attention_mask_k": batch_k["attention_mask"],
            "return_loss": True,
        }
        return batch


# # # Define the metric that we'll use for validation.
# # accuracy = evaluate.load("accuracy")


# # def compute_metrics(eval_pred):
# #     predictions, _ = eval_pred
# #     # Here, predictions is rewards_j and rewards_k.
# #     # We want to see how much of the time rewards_j > rewards_k.
# #     predictions = np.argmax(predictions, axis=0)
# #     labels = np.zeros(predictions.shape)
# #     return accuracy.compute(predictions=predictions, references=labels)


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

output_dir='reward_model_ckpt'
ds_config = "config/ds_config_zero3.json"
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    num_train_epochs=1,
    weight_decay=0,
    evaluation_strategy='no',
    save_strategy="epoch",
    gradient_accumulation_steps=8,
    remove_unused_columns=True,
    deepspeed=ds_config,
    label_names=[],
    fp16=True,
    logging_strategy="steps",
    logging_steps=1,
    lr_scheduler_type="cosine",
)

# Train the model, woohoo.
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,
    # compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=max_length),
)

trainer.train()
print("Saving last checkpoint of the model")
model.save_pretrained(output_dir)
