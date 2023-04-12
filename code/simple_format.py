import json
from transformers import LlamaTokenizer
import click

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

@click.command()
@click.option('-src_ref')
def main(src_ref):
    data = json.load(open(f'filter_{src_ref}_based_data_april_6.json'))
    final_dict = {'type': "text2text", "instances": []}
    len_ls = []
    for _, cur_data in data.items():
        cur_len=len(tokenizer.tokenize(cur_data['input']+cur_data['output']))
        len_ls+=[cur_len]
        if cur_len>682:
            print(cur_len)
        final_dict["instances"]+=[cur_data]

    print(max(len_ls))
    with open(f'llama_{src_ref}_data.json', 'w') as f:
        json.dump(final_dict, f)

if __name__ == "__main__":
    main()
