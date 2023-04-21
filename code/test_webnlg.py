import click
import json
import torch
import os
from tqdm.auto import tqdm
from typing import Optional, Dict, Sequence
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
import glob

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 512
batch_size=1
padding_strategy = "right"
print("Max source length: ", MAX_SOURCE_LENGTH)
print("MAX target length: ", MAX_TARGET_LENGTH)
print("Batch Size: ", batch_size)
print("Padding Strategy: ", padding_strategy)

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

def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:min(i+batch_size, len(lst))]

tokenizer = LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf', model_max_length=MAX_SOURCE_LENGTH, padding_side=padding_strategy, use_fast=False)
smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        tokenizer=tokenizer,
    )

print("Vocab Size: ", len(tokenizer))

@click.command()
@click.option('-sys_name')
@click.option('-task_name')
@click.option('-loaded', type=bool, default=False)
@click.option('-ckpt_addr', help="LLama_finetune_april_8/checkpoint-148", default=None)
@click.option('-start_index', type=int)
@click.option('-end_index', type=int)
def main(loaded, task_name, sys_name, ckpt_addr, start_index, end_index):
    if not loaded:
        if not os.path.isdir(f'test_{task_name}'):
            os.makedirs(f'test_{task_name}')
            os.makedirs(f'test_{task_name}/ref')
            index = ckpt_addr.split('-')[-1]
            print(index)
            os.makedirs(f'test_{task_name}/output_{index}')
        
        score_data = json.load(open('challenge-2020/evaluation/human-evaluation/results/en/english_humeval_data_all_teams.json'))
        index_set = set() 
        gt_dict = {}
        for data in score_data:
            index_set.add(data["sample_id"])
            if data["submission_id"] not in gt_dict:
                gt_dict[data["submission_id"]] = {}
            if data["sample_id"] not in gt_dict[data["submission_id"]]:
                gt_dict[data["submission_id"]][data["sample_id"]] = {}
            gt_dict[data["submission_id"]][data["sample_id"]]["Correctness"] = data["Correctness"]
            gt_dict[data["submission_id"]][data["sample_id"]]["DataCoverage"] = data["DataCoverage"]
            gt_dict[data["submission_id"]][data["sample_id"]]["Fluency"] = data["Fluency"]
            gt_dict[data["submission_id"]][data["sample_id"]]["Relevance"] = data["Relevance"]
            gt_dict[data["submission_id"]][data["sample_id"]]["TextStructure"] = data["TextStructure"]
        
        index_set-={'1124'}

        ref_data = json.load(open('challenge-2020/evaluation/references/references-en.json'))
        
        print("Num of sys: ", len(glob.glob('challenge-2020/submissions/rdf2text/en/*')))
        for file_name in glob.glob('challenge-2020/submissions/rdf2text/en/*'):
            cur_sys = file_name.split('/')[-1]
            final_ref_dict = {'type': "text2score", "instances": []}
            lines = open(file_name+'/primary.en', 'r').readlines()

            for index in index_set:
                ref = ref_data['entries'][int(index)-1][index]['lexicalisations'][0]['lex']
                output = lines[int(index)-1]
                ref_prompt=f"You are evaluating Chinese-to-English Machine translation task. The correct translation is \"{ref}\". The model generated translation is \"{output}\". Please identify all errors within each model output, up to a maximum of five. For each error, please give me the corresponding error type, major/minor label, error location of the model generated translation and explanation for the error. Major errors can confuse or mislead the reader due to significant change in meaning, while minor errors don't lead to loss of meaning but will be noticed."
                final_ref_dict["instances"]+=[{"input": ref_prompt, "output": gt_dict[cur_sys][index]}]
            
            with open(f'test_{task_name}/ref/test_{task_name}_{cur_sys}_llama_ref_data.json', 'w') as f:
                json.dump(final_ref_dict, f)
        print("All files are saved!")
    else:
        KEY_TYPE = "type"
        KEY_INSTANCES = "instances"
        # sanity check over the fields of json file
        with open(f'test_{task_name}/ref/test_{task_name}_{sys_name}_llama_ref_data.json') as fin:
            json_data = json.load(fin)
            if KEY_TYPE not in json_data.keys():
                raise ValueError(
                    f'"{KEY_TYPE}" field must be specified for data, e.g.'
                    '{\n'
                    f'   "{KEY_TYPE}: "text2text",\n'
                    f'   "{KEY_INSTANCES}": [\n'
                    '       { "text": "Sentence 1: This is a sentence." }\n'
                    '       { "text": "Sentence 2: This is another sentence." }\n'
                    f'   ]\n'
                    '}'
                )
        device_id = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
        model = LlamaForCausalLM.from_pretrained(ckpt_addr).to(device_id)
        model.eval()
    
        print("Loaded in model and tokenizer!")
        
        index = ckpt_addr.split('-')[-1]
        save_file = open(f'test_{task_name}/output_{index}/test_{task_name}_{sys_name}_llama_ref_data_{start_index}_{end_index}.txt', 'w')

        global_step=0
        with torch.no_grad():
            with tqdm(total=len(json_data["instances"][start_index:end_index])) as pbar:
                for txts_dict in batchify(json_data["instances"][start_index:end_index], batch_size):
                    batch_txts = [txt_dict["input"] for txt_dict in txts_dict]
                    input_ids = tokenizer(batch_txts[0], return_tensors="pt").input_ids.to(device_id)
                    outputs = model.generate(input_ids, max_new_tokens=MAX_TARGET_LENGTH)
                    batch_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    save_file.write(str(global_step+start_index)+'\t'+batch_outputs[0]+'[SEP_WENDA]')
                    global_step+=1
                    pbar.update(1)

        print("File is saved!")

if __name__ == "__main__":
    main()  