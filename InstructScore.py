import torch
from typing import Dict
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 512
print("Max source length: ", MAX_SOURCE_LENGTH)
print("MAX target length: ", MAX_TARGET_LENGTH)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
):
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        }
    )


device_id = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class InstructScore:
    def __init__(self):
        self.tokenizer = LlamaTokenizer.from_pretrained(
            "xu1998hz/InstructScore", model_max_length=MAX_SOURCE_LENGTH, use_fast=False
        )
        # enable batch inference by left padding
        self.tokenizer.padding_side = "left"

        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=self.tokenizer,
        )
        self.model = LlamaForCausalLM.from_pretrained("xu1998hz/InstructScore").to(
            device_id
        )
        self.model.eval()

    def score(self, ref_ls, out_ls):
        prompt_ls = [
            f'You are evaluating Chinese-to-English Machine translation task. The correct translation is "{ref}". The model generated translation is "{out}". Please identify all errors within each model output, up to a maximum of five. For each error, please give me the corresponding error type, major/minor label, error location of the model generated translation and explanation for the error. Major errors can confuse or mislead the reader due to significant change in meaning, while minor\
         errors don\'t lead to loss of meaning but will be noticed.'
            for ref, out in zip(ref_ls, out_ls)
        ]

        with torch.no_grad():
            inputs = self.tokenizer(
                prompt_ls,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SOURCE_LENGTH,
            )
            outputs = self.model.generate(
                inputs["input_ids"].to(device_id),
                attention_mask=inputs["attention_mask"].to(device_id),
                max_new_tokens=MAX_TARGET_LENGTH,
            )
            batch_outputs = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            scores_ls = [
                (-1) * output.count("Major/minor: Minor")
                + (-5) * output.count("Major/minor: Major")
                for output in batch_outputs
            ]
            return batch_outputs, scores_ls


def main():
    refs = [
        "SEScore is a simple but effective next generation text generation evaluation metric",
        "SEScore it really works",
    ]
    outs = [
        "SEScore is a simple effective text evaluation metric for next generation",
        "SEScore is not working",
    ]

    scorer = InstructScore()
    batch_outputs, scores_ls = scorer.score(refs, outs)
    print(batch_outputs)
    print(scores_ls)


if __name__ == "__main__":
    main()
