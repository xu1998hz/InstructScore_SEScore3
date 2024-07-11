import torch
from typing import Dict, TypeVar, Iterable, List
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
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
    def __init__(self, batch_size=2, max_src_len=512, max_trg_len=512, device_id="cuda", task_type="mt_zh-en", cache_dir=None):
        self.batch_size = batch_size
        self.max_src_len = max_src_len
        self.max_trg_len = max_trg_len
        self.device_id = device_id
        self.task_type = task_type
        print("Max source length: ", max_src_len)
        print("MAX target length: ", max_trg_len)

        print("Loading InstructScore model and tokenizer... ")
        if task_type == 'mt_zh-en':
            self.tokenizer = LlamaTokenizer.from_pretrained(
                "xu1998hz/InstructScore", cache_dir=cache_dir, model_max_length=max_src_len, use_fast=False
            )
            self.model = LlamaForCausalLM.from_pretrained("xu1998hz/InstructScore", cache_dir=cache_dir, torch_dtype=torch.bfloat16, device_map="auto")
        elif task_type == "mt_en-es":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "xu1998hz/instructscore_en-es", cache_dir=cache_dir, model_max_length=max_src_len, use_fast=False
            )
            self.model = AutoModelForCausalLM.from_pretrained("xu1998hz/instructscore_en-es", cache_dir=cache_dir, torch_dtype=torch.bfloat16, device_map="auto")
        elif task_type == 'mt_en-ru':
            self.tokenizer = LlamaTokenizer.from_pretrained(
                "xu1998hz/InstructScore", cache_dir=cache_dir, model_max_length=max_src_len, use_fast=False
            )
            self.model = LlamaForCausalLM.from_pretrained("xu1998hz/instructscore_en-ru", cache_dir=cache_dir, torch_dtype=torch.bfloat16, device_map="auto").to(
                device_id
            )
        elif task_type == 'mt_en-de':
            self.tokenizer = LlamaTokenizer.from_pretrained(
                "xu1998hz/InstructScore", cache_dir=cache_dir, model_max_length=max_src_len, use_fast=False
            )
            self.model = LlamaForCausalLM.from_pretrained("xu1998hz/instructscore_en-de", cache_dir=cache_dir, torch_dtype=torch.bfloat16, device_map="auto").to(
                device_id
            )
        elif task_type == 'caption':
            self.tokenizer = LlamaTokenizer.from_pretrained(
                "xu1998hz/InstructScore", cache_dir=cache_dir, model_max_length=max_src_len, use_fast=False
            )
            self.model = LlamaForCausalLM.from_pretrained("xu1998hz/instructscore_caption", cache_dir=cache_dir, torch_dtype=torch.bfloat16, device_map="auto")
        
        elif self.task_type == 'd2t':
            self.tokenizer = LlamaTokenizer.from_pretrained(
                "xu1998hz/InstructScore", cache_dir=cache_dir, model_max_length=max_src_len, use_fast=False
            )
            self.model = LlamaForCausalLM.from_pretrained("xu1998hz/instructscore_data2text", cache_dir=cache_dir, torch_dtype=torch.bfloat16, device_map="auto")
            
        elif self.task_type == 'commonsense':
            self.tokenizer = LlamaTokenizer.from_pretrained(
                "xu1998hz/InstructScore", cache_dir=cache_dir, model_max_length=max_src_len, use_fast=False
            )
            self.model = LlamaForCausalLM.from_pretrained("xu1998hz/instructscore_commonsense", cache_dir=cache_dir, torch_dtype=torch.bfloat16, device_map="auto")

        else:
            print("Task weights are not supported!")
            exit(1)

        self.tokenizer.padding_side = "left"
        # enable batch inference by left padding
        if self.task_type != "mt_en-es":
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=self.tokenizer,
            )
        else:
            self.tokenizer.pad_token = self.tokenizer.eos_token    
        self.model.eval()

    def score(self, ref_ls, out_ls, src_ls=None):
        assert len(ref_ls) == len(out_ls), "The number of references and outputs should be the same."
        if len(ref_ls) == 0 or len(out_ls) == 0:
            return [], []

        if isinstance(ref_ls, str):
            ref_ls = [ref_ls]
        if isinstance(out_ls, str):
            out_ls = [out_ls]
        if isinstance(src_ls, str):
            src_ls = [src_ls]
        
        if self.task_type == 'mt_zh-en':
            prompt_ls = [
                f'You are evaluating Chinese-to-English Machine translation task. The correct translation is "{ref}". The model generated translation is "{out}". Please identify all errors within each model output, up to a maximum of five. For each error, please give me the corresponding error type, major/minor label, error location of the model generated translation and explanation for the error. Major errors can confuse or mislead the reader due to significant change in meaning, while minor errors don\'t lead to loss of meaning but will be noticed.'
                for ref, out in zip(ref_ls, out_ls)
            ]
        elif self.task_type == 'mt_en-de':
            prompt_ls = [
                f'You are evaluating English-to-German Machine translation task. The correct translation is "{ref}". The model generated translation is "{out}". Please identify all errors within each model output, up to a maximum of five. For each error, please give me the corresponding error type, major/minor label, error location of the model generated translation and explanation for the error. Major errors can confuse or mislead the reader due to significant change in meaning, while minor errors don\'t lead to loss of meaning but will be noticed.'
                for ref, out in zip(ref_ls, out_ls)
            ]
        elif self.task_type == 'mt_en-ru':
            prompt_ls = [
                f'You are evaluating English-to-Russian Machine translation task. The correct translation is "{ref}". The model generated translation is "{out}". Please identify all errors within each model output, up to a maximum of five. For each error, please give me the corresponding error type, major/minor label, error location of the model generated translation and explanation for the error. Major errors can confuse or mislead the reader due to significant change in meaning, while minor errors don\'t lead to loss of meaning but will be noticed.'
                for ref, out in zip(ref_ls, out_ls)
            ]
        elif self.task_type == 'mt_en-es':
            prompt_ls = [
                f'You are evaluating English-to-Spanish Machine translation task. The correct translation is "{ref}". The model generated translation is "{out}". Please identify all errors within each model output, up to a maximum of five. For each error, please give me the corresponding error type, major/minor label, error location of the model generated translation and explanation for the error. Major errors can confuse or mislead the reader due to significant change in meaning, while minor errors don\'t lead to loss of meaning but will be noticed.'
                for ref, out in zip(ref_ls, out_ls)
            ]
        elif self.task_type == 'caption':
            prompt_ls = [
                f"""You are evaluating image captioning. The correct generation is "{ref}". The model generated translation is "{out}". Please identify all errors within each model output, up to a maximum of five. For each error, please give me the corresponding error type, major/minor label, error location of the model generated translation and explanation for the error. Major errors can confuse or mislead the reader due to significant change in meaning, while minor errors don't lead to loss of meaning but will be noticed."""
                for ref, out in zip(ref_ls, out_ls)
            ]
        elif self.task_type == 'd2t':
            prompt_ls = [
                f"""You are evaluating RDF-to-text task. The correct generation is "{ref}". The input of model is "{src}". The model generated output is "{out}\n". Please identify all errors within each model output, up to a maximum of five. For each error, please give me the corresponding error dimension, error type, major/minor label, error location of the model generated output and explanation for the error. Major errors can confuse or mislead the reader due to significant change in meaning, while minor errors don't lead to loss of meaning but will be noticed."""
                for ref, out, src in zip(ref_ls, out_ls, src_ls)
            ]
        elif self.task_type == 'keyword-to-text':
            prompt_ls = [
                f"""You are evaluating RDF-to-text task. The correct generation is "{ref}". The input of model is "{src}". The model generated output is "{out}\n". Please identify all errors within each model output, up to a maximum of five. For each error, please give me the corresponding error dimension, error type, major/minor label, error location of the model generated output and explanation for the error. Major errors can confuse or mislead the reader due to significant change in meaning, while minor errors don't lead to loss of meaning but will be noticed."""
                for ref, out, src in zip(ref_ls, out_ls, src_ls)
            ]
        elif self.task_type == 'commonsense':
            prompt_ls = [
                f"""You are evaluating commonsense text generation. The input of model is "{src}". One of the correct generations is "{ref}". The model generated output is "{out}". Please identify all errors within each model output, up to a maximum of five. For each error, please give me the corresponding error type, major/minor label, error location of the model generated output and explanation for the error. Major errors can confuse or mislead the reader due to significant change in meaning, while minor errors don't lead to loss of meaning but will be noticed."""
                for ref, out, src in zip(ref_ls, out_ls, src_ls)
            ]
        else:
            print("Other task type is not supported at moment!")
            exit(1)

        with torch.no_grad():
            batch_outputs_all = []
            scores_ls_all = []
            with tqdm(total=len(prompt_ls)) as pbar:
                for prompt_batch in batchify(prompt_ls, self.batch_size):
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
                        do_sample=False, 
                        temperature=0
                    )
                    batch_outputs = self.tokenizer.batch_decode(
                        outputs[:, inputs.input_ids.shape[1]:],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
                    scores_ls = [
                        (-1) * output.count("Minor\n")
                        + (-5) * output.count("Major\n")
                        for output in batch_outputs
                    ]
                    batch_outputs_all.extend(batch_outputs)
                    scores_ls_all.extend(scores_ls)
                    pbar.update(len(batch_outputs))
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

    task_type="mt_en-es"
    refs=["Y hay una distinción muy importante allí que veremos."]
    outs=["Y hay una distinción muy anormal allí que falta veremos."]

    scorer = InstructScore(device_id=device_id, task_type=task_type, batch_size=6, cache_dir=None)
    if task_type=="commonsense" or task_type=="d2t" or task_type == "key-to-text":
        batch_outputs, scores_ls = scorer.score(ref_ls=refs, out_ls=outs, src_ls=srcs)
    else:
        batch_outputs, scores_ls = scorer.score(ref_ls=refs, out_ls=outs)
    print(batch_outputs)
    print(scores_ls)


if __name__ == "__main__":
    main()
