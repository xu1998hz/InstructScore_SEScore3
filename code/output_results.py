import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from typing import Optional, Dict, Sequence
import transformers
import click

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

ref="""According to au123.com, after a marathon debate and weeks of protests, New South Wales in Australia passed a bill to decriminalize abortion on September 26th (local time).”"""
cand=""""According to the Australian Network, after marathon debate and weeks of protests, on September 26 local time, the new state of Australia passed a bill to legalize abortion."""

txt=\
f"You are evaluating Chinese-to-English Machine translation task. The correct translation is \"{ref}\". The model generated translation is \"{cand}\". Please identify all errors within each model output, up to a maximum of five. For each error, please give me the corresponding error type, major/minor label, error location of the model generated translation and explanation for the error. Major errors can confuse or mislead the reader due to significant change in meaning, while minor errors don't lead to loss of meaning but will be noticed."

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
):
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        }
    )

@click.command()
@click.option('-ckpt_addr', help="LLama_finetune_april_8/checkpoint-148", default=None)
def main(ckpt_addr):
    device_id = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = LlamaForCausalLM.from_pretrained(ckpt_addr).to(device_id)
    model.eval()

    tokenizer = LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf')
    smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
        )

    print("Vocab Size: ", len(tokenizer))
    print("Loaded in model and tokenizer!")

    with torch.no_grad():
        input_ids = tokenizer(txt, return_tensors="pt").input_ids.to(device_id)
        outputs = model.generate(input_ids, max_new_tokens=702)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))

if __name__ == "__main__":
    main()