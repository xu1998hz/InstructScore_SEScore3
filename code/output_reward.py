from transformers import AutoTokenizer, pipeline
from accelerate import Accelerator
import json
from typing import Dict
import transformers
import json
import click

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

current_device = Accelerator().local_process_index
sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 12,
    "truncation": True,
}

@click.command()
@click.option("-ckpt_index", type=int)
@click.option("-sys_name", type=str)
def main(ckpt_index, sys_name):
    f = f'test_wmt22_zh-en/SEScore3_output_sample_222/test_wmt22_zh-en_{sys_name}_llama_ref_data_0_None.txt'
    lines = open(f, 'r').readlines()
    final_str = ''
    for line in lines:
        final_str+=line

    prompt_ls = [ele.split('\t')[-1] for ele in final_str.split('[SEP_WENDA]')[:-1]]
    reward_model_name = f"reward_model_weights/checkpoint-{ckpt_index}"
    tokenizer = AutoTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        tokenizer=tokenizer,
    )

    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=reward_model_name,
        device_map={"": current_device},
        tokenizer=tokenizer,
    )

    pipe_outputs = sentiment_pipe(prompt_ls, **sent_kwargs)
    with open(f"save_{sys_name}_scores_{ckpt_index}.json", "w") as f:
        json.dump({f"{sys_name}": [ele[0]["score"] for ele in pipe_outputs]}, f)

if __name__ == "__main__":
    main()
