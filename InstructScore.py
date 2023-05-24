import torch
from typing import Dict, TypeVar, Iterable, List
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm

T = TypeVar('T')

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


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


class InstructScore:
    def __init__(self, batch_size=2, max_src_len=512, max_trg_len=512, device_id="cuda"):
        self.batch_size = batch_size
        self.max_src_len = max_src_len
        self.max_trg_len = max_trg_len
        self.device_id = device_id
        print("Max source length: ", max_src_len)
        print("MAX target length: ", max_trg_len)

        print("Loading InstructScore model and tokenizer... ")
        self.tokenizer = LlamaTokenizer.from_pretrained(
            "xu1998hz/InstructScore", model_max_length=max_src_len, use_fast=False
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
        assert len(ref_ls) == len(out_ls), "The number of references and outputs should be the same."
        if len(ref_ls) == 0 or len(out_ls) == 0:
            return [], []

        if isinstance(ref_ls, str):
            ref_ls = [ref_ls]
        if isinstance(out_ls, str):
            out_ls = [out_ls]

        prompt_ls = [
            f'You are evaluating Chinese-to-English Machine translation task. The correct translation is "{ref}". The model generated translation is "{out}". Please identify all errors within each model output, up to a maximum of five. For each error, please give me the corresponding error type, major/minor label, error location of the model generated translation and explanation for the error. Major errors can confuse or mislead the reader due to significant change in meaning, while minor\
         errors don\'t lead to loss of meaning but will be noticed.'
            for ref, out in zip(ref_ls, out_ls)
        ]

        with torch.no_grad():
            batch_outputs_all = []
            scores_ls_all = []
            for prompt_batch in tqdm(batchify(prompt_ls, self.batch_size)):
                inputs = self.tokenizer(
                    prompt_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_src_len,
                )
                outputs = self.model.generate(
                    inputs["input_ids"].to(self.device_id),
                    attention_mask=inputs["attention_mask"].to(self.device_id),
                    max_new_tokens=self.max_trg_len,
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
                batch_outputs_all.extend(batch_outputs)
                scores_ls_all.extend(scores_ls)
            return batch_outputs_all, scores_ls_all


def batchify(data: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    assert batch_size > 0

    batch = []
    for item in data:
        # Yield next batch
        if len(batch) == batch_size:
            yield batch
            batch = []

        batch.append(item)

    # Yield last un-filled batch
    if len(batch) != 0:
        yield batch  


def main():
    device_id = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    refs = [
        "SEScore is a simple but effective next generation text generation evaluation metric",
        "SEScore it really works",
    ]
    outs = [
        "SEScore is a simple effective text evaluation metric for next generation",
        "SEScore is not working",
    ]

    scorer = InstructScore(device_id=device_id)
    batch_outputs, scores_ls = scorer.score(refs, outs)
    print(batch_outputs)
    print(scores_ls)


if __name__ == "__main__":
    main()
