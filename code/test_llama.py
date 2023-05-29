import click
import json
import torch
import os
from tqdm.auto import tqdm
from typing import Optional, Dict, Sequence
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
import time

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


def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i : min(i + batch_size, len(lst))]


tokenizer = LlamaTokenizer.from_pretrained(
    "decapoda-research/llama-7b-hf", model_max_length=MAX_SOURCE_LENGTH, use_fast=False
)
# enable batch inference by left padding
tokenizer.padding_side = "left"


@click.command()
@click.option("-wmt")
@click.option("-lang")
@click.option("-sys_name")
@click.option("-src_ref", help="src or ref")
@click.option("-loaded", type=bool, default=False)
@click.option("-ckpt_addr", help="LLama_finetune_april_8/checkpoint-148", default=None)
@click.option("-start_index", type=int)
@click.option("-end_index", type=int)
@click.option("-batch_size", type=int)
@click.option("-sample", type=bool)
@click.option("-num_ret", type=int)
@click.option("-task_mode", type=str, help="evaluation or machine translation")
def main(
    wmt,
    lang,
    loaded,
    src_ref,
    sys_name,
    ckpt_addr,
    start_index,
    end_index,
    batch_size,
    sample,
    num_ret,
    task_mode
):
    if not loaded:
        if lang == "zh-en":
            lang_code = "Chinese-to-English"
        elif lang == "en-de":
            lang_code = "English-to-German"
        elif lang == "en-ru":
            lang_code = "English-to-Russian"
        else:
            print("Language dir is not existed")
            exit(1)

        from mt_metrics_eval import data

        if not os.path.isdir(f"test_{wmt}_{lang}"):
            os.makedirs(f"test_{wmt}_{lang}")
            os.makedirs(f"test_{wmt}_{lang}/src")
            os.makedirs(f"test_{wmt}_{lang}/ref")
            index = ckpt_addr.split("-")[-1]
            print(index)
            os.makedirs(f"test_{wmt}_{lang}/SEScore3_output_{index}")

        evs = data.EvalSet(wmt, lang)
        mqm_scores = evs.Scores("seg", "mqm")
        print("ref: ", evs.std_ref)
        print("Annotated System: ", len(mqm_scores))

        for sys_name, score_ls in mqm_scores.items():
            assert len(score_ls) == len(evs.sys_outputs[sys_name])
            if sys_name != evs.std_ref:
                final_ref_dict = {"type": "text2score", "instances": []}
                final_src_dict = {"type": "text2score", "instances": []}
                for index, (score, output) in enumerate(
                    zip(score_ls, evs.sys_outputs[sys_name])
                ):
                    if sys_name != evs.std_ref:
                        if score != None:
                            ref = evs.sys_outputs[evs.std_ref][index]
                            cand = evs.sys_outputs[sys_name][index]
                            src = evs.src[index]
                            ref_prompt = f'You are evaluating {lang_code} Machine translation task. The correct translation is "{ref}". The model generated translation is "{cand}". Please identify all errors within each model output, up to a maximum of five. For each error, please give me the corresponding error type, major/minor label, error location of the model generated translation and explanation for the error. Major errors can confuse or mislead the reader due to significant change in meaning, while minor errors don\'t lead to loss of meaning but will be noticed.'
                            src_prompt = f'You are evaluating {lang_code} Machine translation task. The source is "{src}". The model generated translation is "{cand}". Please identify all errors within each model output, up to a maximum of five. For each error, please give me the corresponding error type, major/minor label, error location of the model generated translation and explanation for the error. Major errors can confuse or mislead the reader due to significant change in meaning, while minor errors don\'t lead to loss of meaning but will be noticed.'

                            final_ref_dict["instances"] += [
                                {"input": ref_prompt, "output": score}
                            ]
                            final_src_dict["instances"] += [
                                {"input": src_prompt, "output": score}
                            ]

            if len(final_ref_dict["instances"]) > 0:
                with open(
                    f"test_{wmt}_{lang}/ref/test_{wmt}_{lang}_{sys_name}_llama_ref_data.json",
                    "w",
                ) as f:
                    json.dump(final_ref_dict, f)

            if len(final_src_dict["instances"]) > 0:
                with open(
                    f"test_{wmt}_{lang}/src/test_{wmt}_{lang}_{sys_name}_llama_src_data.json",
                    "w",
                ) as f:
                    json.dump(final_src_dict, f)
                print(f"test_{wmt}_{lang}_{sys_name} ref and src files are saved!")
    else:
        if task_mode == 'evaluation':
            KEY_TYPE = "type"
            KEY_INSTANCES = "instances"
            # sanity check over the fields of json file
            with open(
                f"test_{wmt}_{lang}/{src_ref}/test_{wmt}_{lang}_{sys_name}_llama_{src_ref}_data.json"
            ) as fin:
                json_data = json.load(fin)
                if KEY_TYPE not in json_data.keys():
                    raise ValueError(
                        f'"{KEY_TYPE}" field must be specified for data, e.g.'
                        "{\n"
                        f'   "{KEY_TYPE}: "text2text",\n'
                        f'   "{KEY_INSTANCES}": [\n'
                        '       { "text": "Sentence 1: This is a sentence." }\n'
                        '       { "text": "Sentence 2: This is another sentence." }\n'
                        f"   ]\n"
                        "}"
                    )
            device_id = (
                torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            )
            model = LlamaForCausalLM.from_pretrained(ckpt_addr).to(device_id)
            model.eval()

            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
            )

            print("Vocab Size: ", len(tokenizer))
            print("Loaded in model and tokenizer!")

            index = ckpt_addr.split("-")[-1]
            if sample:
                save_file = open(
                    f"test_{wmt}_{lang}/SEScore3_output_sample_{index}/test_{wmt}_{lang}_{sys_name}_llama_{src_ref}_data_{start_index}_{end_index}_sample_{num_ret}.txt",
                    "w",
                )
            else:
                save_file = open(
                    f"test_{wmt}_{lang}/SEScore3_output_{index}/test_{wmt}_{lang}_{sys_name}_llama_{src_ref}_data_{start_index}_{end_index}.txt",
                    "w",
                )

            global_step = 0
            with torch.no_grad():
                with tqdm(
                    total=int(
                        len(json_data["instances"][start_index:end_index]) / batch_size
                    )
                    + 1
                ) as pbar:
                    for txts_dict in batchify(
                        json_data["instances"][start_index:end_index], batch_size
                    ):
                        batch_txts = [txt_dict["input"] for txt_dict in txts_dict]
                        inputs = tokenizer(
                            batch_txts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=MAX_SOURCE_LENGTH,
                        )
                        try:
                            if sample:
                                outputs = model.generate(
                                    inputs["input_ids"].to(device_id),
                                    attention_mask=inputs["attention_mask"].to(device_id),
                                    max_new_tokens=MAX_TARGET_LENGTH,
                                    do_sample=True,
                                    top_p=0.95,
                                    temperature=0.8,
                                    num_return_sequences=num_ret,
                                )
                            else:
                                outputs = model.generate(
                                    inputs["input_ids"].to(device_id),
                                    attention_mask=inputs["attention_mask"].to(device_id),
                                    max_new_tokens=MAX_TARGET_LENGTH,
                                )
                            batch_outputs = tokenizer.batch_decode(
                                outputs,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=True,
                            )
                            if sample:
                                for index, output in enumerate(batch_outputs):
                                    save_file.write(
                                        str(global_step + start_index)
                                        + "\t"
                                        + str(index)
                                        + "\t"
                                        + output
                                        + "[SEP_WENDA]"
                                    )
                                global_step += 1
                            else:
                                for output in batch_outputs:
                                    save_file.write(
                                        str(global_step + start_index)
                                        + "\t"
                                        + output
                                        + "[SEP_WENDA]"
                                    )
                                    global_step += 1
                        except Exception as e:
                            print(f"ERROR: {e}")
                        pbar.update(1)

        print("File is saved!")


if __name__ == "__main__":
    main()
