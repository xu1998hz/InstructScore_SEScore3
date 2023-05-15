import ast
import json
import click
import re

""" python3 code/post_process_gpt4_feedback.py -start_str "0 800 1200" -end_str "800 1200 2000" """


@click.command()
@click.option("-start_str", type=str)
@click.option("-end_str", type=str)
def main(start_str, end_str):
    non_json_err_count = 0
    miss_q7_count = 0
    q_num_err_count = 0
    cor_count = 0
    index_set = set()
    final_dict = {}

    prefix = 'The correct translation is "'
    middle_suffix = '". The model generated translation is "'
    suffix = '". Please identify all errors within each model output'

    final_str = ""
    for start_file_index in range(0, 10000, 2000):
        end_file_index = start_file_index + 2000
        out_ref_ls = open(
            f"data/wmt_samples/test_wmt20_zh-en_samples_llama_ref_data_{start_file_index}_{end_file_index}.txt",
            "r",
        ).readlines()
        for ele in out_ref_ls:
            final_str += ele

    ref_out_dict = {}
    pattern = r"Major/minor: (.*?)\n"
    for line in final_str.split("[SEP_WENDA]")[:-1]:
        cur_index, prompt_str = int(line.split("\t")[0]), line.split("\t")[1]
        if cur_index not in ref_out_dict:
            ref_out_dict[cur_index] = []
        ref = prompt_str.split(prefix)[1].split(middle_suffix)[0]
        out = prompt_str.split(middle_suffix)[1].split(suffix)[0]
        result = re.findall(pattern, prompt_str)
        result = [ele.lower() + "-error" for ele in result]
        ref_out_dict[cur_index] += [
            {"ref": ref, "out": out, "labels": result, "prompt": prompt_str}
        ]

    for start_index, end_index in zip(
        [int(ele) for ele in start_str.split()], [int(ele) for ele in end_str.split()]
    ):
        indicater_lines = open(
            f"reward_data/fixed_new_reward_samples_indicater_{start_index}_{end_index}.txt"
        ).readlines()
        final_str_ind = ""
        for ele in indicater_lines:
            final_str_ind += ele

        lines = open(
            f"reward_data/fixed_new_reward_samples_{start_index}_{end_index}.txt"
        ).readlines()
        final_str = ""
        for ele in lines:
            final_str += ele

        assert len(final_str.split("[SEP_TOK_WENDA]")) == len(
            final_str_ind.split("[SEP_TOK_WENDA]")
        )

        for ele, ele_ind in zip(
            final_str.split("[SEP_TOK_WENDA]")[:-1],
            final_str_ind.split("[SEP_TOK_WENDA]")[:-1],
        ):
            cur_ind = ele_ind.split("\t")[-1] == "True"
            cur_index, num_seq = int(ele.split("\t")[0]), int(ele.split("\t")[1])

            if cur_index not in final_dict:
                final_dict[cur_index] = {}

            if cur_ind:
                loaded_dict = {}
                loaded_dict["no_error_ind"] = cur_ind
                proc_str = ele.split("\t")[-1]
                if proc_str == "Yes":
                    loaded_dict["no_error_score"] = 0
                else:
                    loaded_dict["no_error_score"] = 1
                cor_count += 1
                loaded_dict["prompt"] = ref_out_dict[cur_index][num_seq]["prompt"]
                final_dict[cur_index][num_seq] = loaded_dict
                index_set.add(cur_index)
            else:
                proc_str = ele.split("\t")[-1].replace("\n", "").replace("  ", "")
                try:
                    loaded_dict = ast.literal_eval(proc_str)
                    # first check if Q7 is in dict
                    if "Q7" in loaded_dict:
                        q_num_check = True
                        for err in loaded_dict:
                            # check each err fields
                            if err != "Q7":
                                if len(loaded_dict[err]) != 6:
                                    q_num_check = False
                                    break
                                else:
                                    if (
                                        loaded_dict[err]["Q1"] != "A1"
                                        and len(loaded_dict[err]["Q2"]) != 2
                                        and (
                                            loaded_dict[err]["Q3"].lower() == "yes"
                                            or loaded_dict[err]["Q3"].lower() == "no"
                                        )
                                        and (
                                            loaded_dict[err]["Q4"].lower()
                                            == "major-error"
                                            or loaded_dict[err]["Q4"].lower()
                                            == "minor-error"
                                            or loaded_dict[err]["Q4"].lower()
                                            == "no-error"
                                        )
                                        and (
                                            loaded_dict[err]["Q5"].lower() == "yes"
                                            or loaded_dict[err]["Q5"].lower() == "no"
                                        )
                                        and (
                                            loaded_dict[err]["Q6"].lower() == "yes"
                                            or loaded_dict[err]["Q6"].lower() == "no"
                                        )
                                    ):
                                        q_num_check = False
                                        break
                        if q_num_check:
                            if (
                                len(ref_out_dict[cur_index][num_seq]["labels"])
                                == len(loaded_dict) - 1
                            ):
                                cor_count += 1
                                loaded_dict["ref"] = ref_out_dict[cur_index][num_seq][
                                    "ref"
                                ]
                                loaded_dict["out"] = ref_out_dict[cur_index][num_seq][
                                    "out"
                                ]
                                loaded_dict["labels"] = ref_out_dict[cur_index][
                                    num_seq
                                ]["labels"]
                                loaded_dict["prompt"] = ref_out_dict[cur_index][
                                    num_seq
                                ]["prompt"]
                                loaded_dict["no_error_ind"] = cur_ind
                                final_dict[cur_index][num_seq] = loaded_dict
                                index_set.add(cur_index)
                        else:
                            q_num_err_count += 1
                    else:
                        miss_q7_count += 1
                except Exception as e:
                    # print(e)
                    non_json_err_count += 1

    print("Corect Count: ", cor_count)
    print("Non JSON Error Count: ", non_json_err_count)
    print("Missing Q7: ", miss_q7_count)
    print("Num of missing Err Qs: ", q_num_err_count)
    print("Instance Pass Rate: ", len(index_set) / 2000)
    print(
        "Total Pass Rate (JSON): ",
        cor_count / (non_json_err_count + cor_count + miss_q7_count + q_num_err_count),
    )
    with open("reward_data/final_data_scores.json", "w") as f:
        json.dump(final_dict, f)


if __name__ == "__main__":
    main()
