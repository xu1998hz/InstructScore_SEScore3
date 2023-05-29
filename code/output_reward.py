from transformers import AutoTokenizer, pipeline
from accelerate import Accelerator
import json
from typing import Dict
import transformers
import click
import glob

final_dict = {}
for file_addr in glob.glob("test_wmt22_zh-en/SEScore3_output_sample_222/*"):
    file_name = file_addr.split("/")[-1]
    sys_name = file_name.split("test_wmt22_zh-en_")[1].split("_llama_")[0]
    final_str = ""
    lines = open(file_addr, "r").readlines()
    for ele in lines:
        final_str += ele
    if sys_name not in final_dict:
        final_dict[sys_name] = {}
    for cur_str in final_str.split("[SEP_WENDA]")[:-1]:
        sen_index, ret_index, output = (
            cur_str.split("\t")[0],
            cur_str.split("\t")[1],
            cur_str.split("\t")[2],
        )
        if sen_index not in final_dict[sys_name]:
            final_dict[sys_name][sen_index] = {}
        final_dict[sys_name][sen_index][ret_index] = output

# sys_ls = [
#     "LanguageX",
#     "M2M100_1.2B-B4",
#     "Online-A",
#     "Online-B",
#     "Online-G",
#     "Online-W",
#     "Online-Y",
#     "bleurt_bestmbr",
# ]

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
    "batch_size": 6,
    "truncation": True,
}


@click.command()
@click.option("-sys_name", type=str)
def main(sys_name):
    reward_model_name = f"/share/edc/home/wendaxu/Reward_Model"
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

    output_ls = []
    for ele in list(final_dict[sys_name].values()):
        output_ls += [cur_ele for cur_ele in ele.values()]

    pipe_outputs = sentiment_pipe(output_ls, **sent_kwargs)
    with open(f"save_{sys_name}_scores_3.json", "w") as f:
        json.dump({f"{sys_name}": [ele[0]["score"] for ele in pipe_outputs]}, f)


if __name__ == "__main__":
    main()
