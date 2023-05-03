import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from typing import Optional, Dict, Sequence
import transformers
import click
from typing import List
import time

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
MAX_SEQ_LEN = 720
MAX_TARGET_LENGTH = 512

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

class LLaMA:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        bsz = len(prompts)
        prompt_tokens = [self.tokenizer.encode(x) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(MAX_SEQ_LEN, MAX_TARGET_LENGTH + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.get_vocab()[DEFAULT_PAD_TOKEN]).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.get_vocab()[DEFAULT_PAD_TOKEN]
        start_pos = min_prompt_size
        prev_pos = 0
        cur_status = {i: 0 for i in range(bsz)}
        for cur_pos in range(start_pos, total_len):
            if bsz == sum(cur_status.values()):
                break
            logits = self.model.forward(tokens[:, prev_pos:cur_pos]).logits[:, -1, :]
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            # implement the early step criterion for generation
            if torch.sum(next_token==self.tokenizer.get_vocab()[DEFAULT_EOS_TOKEN])>0:
                for index, status in enumerate(next_token==self.tokenizer.get_vocab()[DEFAULT_EOS_TOKEN]):
                    if status:
                        cur_status[index] = 1
            tokens[:, cur_pos] = next_token

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.get_vocab()[DEFAULT_EOS_TOKEN])]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

@click.command()
@click.option('-ckpt_addr', help="checkpoint-222", default=None)
def main(ckpt_addr):
    device_id = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = LlamaForCausalLM.from_pretrained(ckpt_addr, torch_dtype=torch.float16).to(device_id)
    model.eval()

    tokenizer = LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf')
    smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
        )

    print("Vocab Size: ", len(tokenizer))

    llama = LLaMA(model, tokenizer)
    print("Loaded in model and tokenizer!")

    start_time = time.time()
    with torch.no_grad():
        decoded = llama.generate([txt]*8, max_gen_len=512, temperature=0)
        print(decoded)
    print(time.time()-start_time)

if __name__ == "__main__":
    main()