from datasets import load_dataset
import random
import json
import click

@click.command()
@click.option('-src_code')
@click.option('-tar_code')
def main(src_code, tar_code):
    code_lang_dict = {'en': "English", "de": "German", "ru": "Russian"}

    if f"{tar_code}-{src_code}" == 'de-en':
        wmt = 'wmt14'
        load_addr = 'data/german_llama_ref_data.json'
        save_addr = 'data/eval_mt_german_llama.json'
    elif f"{tar_code}-{src_code}" == 'ru-en':
        wmt = 'wmt16'
        load_addr = 'data/russian_llama_ref_data.json'
        save_addr = 'data/eval_mt_russian_llama.json'
    else:
        print("Language is not supported!")

    data = load_dataset(wmt, f"{tar_code}-{src_code}")
    # extract 2.5% training data, 10% of validation data
    train_data_part1, train_data_part2, vali_data = data['train'][:10000], data['train'][10000:20000], data['validation'][:300]

    cur_data = json.load(open(load_addr))

    for cur_dict in train_data_part1['translation']:
        src_txt, tar_txt = cur_dict[src_code], cur_dict[tar_code]
        src_lang, tar_lang = code_lang_dict[src_code], code_lang_dict[tar_code]
        prompt = f"You are translating from {src_lang} to {tar_lang}. The {src_lang} text is \"{src_txt}\". The {tar_lang} text is "
        cur_data["instances"]+=[{"input": prompt, "output": tar_txt}]

    for cur_dict in train_data_part2['translation']:
        src_txt, tar_txt = cur_dict[tar_code], cur_dict[src_code]
        src_lang, tar_lang = code_lang_dict[tar_code], code_lang_dict[src_code]
        prompt = f"You are translating from {src_lang} to {tar_lang}. The {src_lang} text is \"{src_txt}\". The {tar_lang} text is "
        cur_data["instances"]+=[{"input": prompt, "output": tar_txt}]

    with open(save_addr, 'w') as f:
        json.dump(cur_data, f)

    print("File is saved!")

if __name__ == "__main__":
    main()