import glob
from openai.error import RateLimitError
import backoff
import openai
from tqdm.auto import tqdm
import click
import asyncio
from typing import Any


@backoff.on_exception(backoff.expo, RateLimitError)
async def dispatch_openai_requests(
    messages_list,
    model,
    temperature,
    max_tokens,
    top_p,
):
    """Dispatches requests to OpenAI API asynchronously.

    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


suffix_1 = 'You are evaluating Chinese-to-English Machine translation task. The correct translation is "'
suffix_2 = '". The model generated translation is "'
suffix_3 = '". Please identify all errors'
suffix_4 = "while minor errors don't lead to loss of meaning but will be noticed."

# load in all sample data
cur_dict = {}
for file_name in sorted(glob.glob("data/wmt_samples/*")):
    lines = open(file_name, "r").readlines()
    final_str = "".join(lines)
    samples_ls = final_str.split("[SEP_WENDA]")[:-1]
    for ele in samples_ls:
        if ele.split("\t")[0] not in cur_dict:
            cur_dict[ele.split("\t")[0]] = []
        cur_dict[ele.split("\t")[0]] += [ele.split("\t")[1]]

dist_dict = {}
for index, output in cur_dict.items():
    if len(set(output)) not in dist_dict:
        dist_dict[len(set(output))] = 0
    dist_dict[len(set(output))] += 1


@click.command()
@click.option("-gpt_mode")
@click.option("-api_key")
@click.option("-interval", help="Use 5", type=int)
@click.option("-start_index", help="Use 0", type=int)
@click.option("-end_index", help="Use 400", type=int)
def main(gpt_mode, api_key, interval, start_index, end_index):
    openai.api_key = api_key
    temperature = 0
    count = 0
    max_tokens = 512
    saveFile = open(f"fixed_new_reward_samples_{start_index}_{end_index}.txt", "w")
    saveIndFile = open(
        f"fixed_new_reward_samples_indicater_{start_index}_{end_index}.txt", "w"
    )
    # display all the model outputs + explanations
    index_ls = list(range(0, 10000, interval))[start_index:end_index]
    with tqdm(total=len(index_ls)) as pbar:
        for cur_index in index_ls:
            batch_messages_ls, batch_sample_index_ls, batch_indicater_ls = [], [], []
            for sample_index, ele in enumerate(cur_dict[str(cur_index)]):
                if suffix_4 in ele:
                    score = ele.split(suffix_4)[1].count("Major/minor: Major") * (
                        -5
                    ) + ele.split(suffix_4)[1].count("Major/minor: Minor") * (-1)
                    ref = ele.split(suffix_1)[1].split(suffix_2)[0]
                    out = ele.split(suffix_2)[1].split(suffix_3)[0]
                    if score == 0:
                        # detect if translation is truly no-error
                        sanity_tp_prompt = f"""Reference: {ref}\n Output: {out}\n Compared to the reference, does the output contain any error? (Answer in Yes/No)\n"""
                        batch_messages_ls += [
                            [{"role": "user", "content": sanity_tp_prompt}]
                        ]
                        batch_sample_index_ls += [sample_index]
                        batch_indicater_ls += [True]
                    else:
                        # construct global prompt, to detect overlap or repetition explanations
                        num_errs = int(
                            ele.split(suffix_4)[1]
                            .split("Your Translation contains ")[1]
                            .split("error")[0]
                        )
                        reward_prompt = ""
                        if (
                            f"Error location {num_errs}: " in ele.split(suffix_4)[1]
                            and f"Explanation for error {num_errs}: "
                            in ele.split(suffix_4)[1]
                        ):
                            for i in range(1, 1 + num_errs):
                                err_loc = (
                                    ele.split(suffix_4)[1]
                                    .split(f"Error location {i}: ")[1]
                                    .split("\n")[0]
                                )
                                err_type = (
                                    ele.split(suffix_4)[1]
                                    .split(f"Error type {i}: ")[1]
                                    .split("\n")[0]
                                )
                                reward_prompt += f"Error{i}:\n"
                                reward_prompt += f"Error location {i}: {err_loc}\n"
                                reward_prompt += f"Error type {i}: {err_type}\n"

                                if i < num_errs:
                                    explain = (
                                        ele.split(suffix_4)[1]
                                        .split(f"Explanation for error {i}: ")[1]
                                        .split("\n")[0]
                                    )
                                else:
                                    explain = ele.split(suffix_4)[1].split(
                                        f"Explanation for error {i}: "
                                    )[1]
                                reward_prompt += f"Explanation{i}: {explain}\n"
                            reward_prompt += """\nQ1: For each error location, extract the incorrect error location.\nA1:\n"""
                            reward_prompt += """Q2: Parse explanation into either one of the four forms: [incorrect phrase, correct phrase], [incorrect phrase, None], [None, correct phrase] or [None, None].\nA2:\n"""
                            reward_prompt += f"""\nRef: {ref}\nOut: {out}\n"""
                            reward_prompt += """\nQ3: If A2 is "incorrect phrase to correct phrase", is A2 a correct alignment for reference and output? (Answer Yes/No) If A2 contains None, Answer None.\nA3:\n"""
                            reward_prompt += """Q4: If A2 is "incorrect phrase to correct phrase" and incorrect and correct phrases can be used interchangeably, it is no-error. If an error can be fixed by reading the rest of the output context it is a minor-error. If it changes the sentence's meaning (Don't consider sentence formality changes) and is unable to be fixed by reading the rest of the output it is a major-error. Answer, no-error or minor-error or major-error\nA4:\n"""
                            reward_prompt += """Q5: Is explanation consistent to the given error type?\n"""
                            reward_prompt += """A5: \n"""
                            reward_prompt += """Q6: Does the explanation talk about the given error location?\n"""
                            reward_prompt += """A6: \n"""
                            reward_prompt += """\nQ7: Do two error locations mention the same location in two different ways? (Answer Yes/No) How many pairs of repetitions? (Answer in a number)\nA7: Yes/No, number\n"""
                            reward_prompt += (
                                """\nThe output format will be in JSON\n{"""
                            )
                            cur_ls = []
                            for i in range(1, 1 + num_errs):
                                cur_str = f"Err{i}: "
                                cur_str += (
                                    "{Q1: A1, Q2: A2, Q3: A3, Q4: A4, Q5: A5, Q6: A6}"
                                )
                                cur_ls += [cur_str]
                            cur_ls += ["Q7: A7"]
                            reward_prompt += "{" + (", ").join(cur_ls) + "}"
                            batch_messages_ls += [
                                [{"role": "user", "content": reward_prompt}]
                            ]
                            batch_sample_index_ls += [sample_index]
                            batch_indicater_ls += [False]
                        else:
                            count += 1
                else:
                    count += 1

            try:
                predictions = asyncio.run(
                    dispatch_openai_requests(
                        messages_list=batch_messages_ls,
                        model=gpt_mode,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=1.0,
                    )
                )

                for pred, sample_index, cur_sample_indicater in zip(
                    predictions, batch_sample_index_ls, batch_indicater_ls
                ):
                    saveFile.write(
                        str(cur_index)
                        + "\t"
                        + str(sample_index)
                        + "\t"
                        + pred["choices"][0]["message"]["content"]
                        + "[SEP_TOK_WENDA]"
                    )
                    saveFile.flush()
                    saveIndFile.write(
                        str(cur_index)
                        + "\t"
                        + str(sample_index)
                        + "\t"
                        + str(cur_sample_indicater)
                        + "[SEP_TOK_WENDA]"
                    )
                    saveIndFile.flush()

            except Exception as e:
                print(f"ERROR: {e}")
            pbar.update(1)

    print(count)
    print("All files are saved!")


if __name__ == "__main__":
    main()
