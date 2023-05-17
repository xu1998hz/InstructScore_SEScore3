import json
import ast
import itertools

data = json.load(open("reward_data/final_data_scores.json"))
total = 0
# count different types of errors
num_repeat = 0
explanation_consistency_err = 0
err_loc_err = 0
inconsistency_err_loc = 0
inconsistency_err_type = 0
mis_align_err = 0
output_hallucinations = 0
ref_hallucinations = 0
ref_multiple_err = 0
num_label_conflict = 0
num_no_err_minor_disagreement = 0
num_no_err_major_disagreement = 0
num_minor_major_disagreement = 0
multiple_err_loc = 0
total_instances = 0
total_better_ls, total_worse_ls = [], []

for index, sen_dict in data.items():
    sen_score_ls = []
    for sen_index, feedback_dict in sen_dict.items():
        total_instances += 1
        if feedback_dict["no_error_ind"]:
            final_score = feedback_dict["no_error_score"]
        else:
            # first check the repetition
            neg_score = 0
            if feedback_dict["Q7"].split(",")[0] == "Yes":
                num_repeat += 1
                neg_score = int(feedback_dict["Q7"].split(",")[1].strip()) * 4

            cur_err_dict = {}
            for err_index, (err_key, err_dict) in enumerate(feedback_dict.items()):
                if err_key[:3] == "Err":
                    cur_err_dict[err_key] = {
                        "err_type": 1,
                        "err_loc": 1,
                        "major_minor": 1,
                        "explanation": 1,
                    }
                    total += 1
                    """Check all for major/minor labels"""
                    if feedback_dict["labels"][err_index] != err_dict["Q4"]:
                        cur_err_dict[err_key]["major_minor"] = 0
                        cur_err_dict[err_key]["explanation"] = 0
                        if (
                            err_dict["Q4"] == "no-error"
                            and feedback_dict["labels"][err_index] == "minor-error"
                        ):
                            num_no_err_minor_disagreement += 1
                            cur_err_dict[err_key]["err_type"] = 0
                            cur_err_dict[err_key]["err_loc"] = 0
                        elif (
                            err_dict["Q4"] == "no-error"
                            and feedback_dict["labels"][err_index] == "major-error"
                        ):
                            cur_err_dict[err_key]["err_type"] = 0
                            cur_err_dict[err_key]["err_loc"] = 0
                            num_no_err_major_disagreement += 1
                        else:
                            num_minor_major_disagreement += 1

                    """Check all for Error locations"""
                    # check hallucination cases on error location
                    if err_dict["Q1"] != None or err_dict["Q1"] != "None":
                        if isinstance(err_dict["Q1"], list):
                            if (
                                err_dict["Q1"][0] != None
                                and err_dict["Q1"][0] != "None"
                                and (err_dict["Q1"][0] not in feedback_dict["out"])
                            ):
                                err_loc_err += 1
                                cur_err_dict[err_key]["err_loc"] = 0
                        elif isinstance(err_dict["Q1"], str):
                            if (
                                err_dict["Q1"] != "None"
                                and err_dict["Q1"] not in feedback_dict["out"]
                            ):
                                err_loc_err += 1
                                cur_err_dict[err_key]["err_loc"] = 0

                    """Check all for Explanation"""
                    # check the misalignment between two phrases
                    if err_dict["Q3"] == "No":
                        mis_align_err += 1
                        cur_err_dict[err_key]["explanation"] = 0

                    # check the consistency between error location and explanation
                    print(err_dict["Q6"])
                    if err_dict["Q6"] != "Yes":
                        inconsistency_err_loc += 1
                        cur_err_dict[err_key]["err_loc"] = 0

                    # check the consistency between error type and explanation
                    if err_dict["Q5"] != "Yes":
                        inconsistency_err_type += 1
                        cur_err_dict[err_key]["err_type"] = 0

                    if not isinstance(err_dict["Q2"], list):
                        try:
                            err_dict["Q2"] = ast.literal_eval(err_dict["Q2"])
                        except Exception as e:
                            if (
                                err_dict["Q2"].count("[") == 1
                                and err_dict["Q2"].count("]") == 1
                                and err_dict["Q2"].count(", ") == 1
                            ):
                                err_dict["Q2"] = err_dict["Q2"][1:-1].split(", ")
                            else:
                                # All those cases can be considered as multiple errors skip all later lines
                                multiple_err_loc += 1
                                cur_err_dict[err_key]["err_loc"] = 0
                                cur_err_dict[err_key]["explanation"] = 0
                                continue

                    # check output hallucinations in the explanations
                    if err_dict["Q2"] != None and err_dict["Q2"] != "None":
                        if isinstance(err_dict["Q2"][0], list):
                            # check if there are multiple errors in one explanation
                            multiple_err_loc += 1
                            cur_err_dict[err_key]["err_loc"] = 0
                            cur_err_dict[err_key]["explanation"] = 0
                        else:
                            # check the hallucination on the output in explanation
                            if err_dict["Q2"][0] != None and (
                                err_dict["Q2"][0] != "None"
                                and err_dict["Q2"][0] not in feedback_dict["out"]
                            ):
                                output_hallucinations += 1
                                cur_err_dict[err_key]["explanation"] = 0

                            # check the consistency of the explanation
                            if (err_dict["Q2"][0] == err_dict["Q2"][1]) or (
                                err_dict["Q2"][1] != None
                                and isinstance(err_dict["Q2"], str)
                                and err_dict["Q2"][1] in feedback_dict["out"]
                            ):
                                explanation_consistency_err += 1
                                cur_err_dict[err_key]["explanation"] = 0

            cur_score = 0
            total_score = len(cur_err_dict) * 4
            for err_key, score_dict in cur_err_dict.items():
                cur_score += sum(list(score_dict.values()))
            final_score = (cur_score - neg_score) / total_score
        sen_score_ls += [[feedback_dict["prompt"], final_score]]
    better_ls, worse_ls = [], []
    for pair in list(itertools.combinations(sen_score_ls, 2)):
        if pair[0][1] > pair[1][1]:
            better_ls += [pair[0][0]]
            worse_ls += [pair[1][0]]
        elif pair[0][1] < pair[1][1]:
            better_ls += [pair[1][0]]
            worse_ls += [pair[0][0]]
    total_better_ls += better_ls
    total_worse_ls += worse_ls

print("Better: ", len(total_better_ls))
print("Worse: ", len(total_worse_ls))
save_final_dict = {
    "type": "text2text",
    "instances": [
        {"response_j": ele_better, "response_k": ele_worse}
        for ele_better, ele_worse in zip(total_better_ls, total_worse_ls)
    ],
}
with open("final_pairwise_data.json", "w") as f:
    json.dump(save_final_dict, f)
print("File is saved!")
print("Error location hallucination error: ", err_loc_err / total)
print("Error location inconsistency error: ", inconsistency_err_loc / total)
print("Error type inconsistency error: ", inconsistency_err_type / total)
print(
    "Major vs minor disagreement errors in explanation: ",
    num_minor_major_disagreement / total,
)
print(
    "no-error vs minor disagreement errors in explanation: ",
    num_no_err_minor_disagreement / total,
)
print(
    "no-error vs major disagreement errors in explanation: ",
    num_no_err_major_disagreement / total,
)
print("Misligned Error Rate: ", mis_align_err / total)
print("Explanation Output hallucinations: ", output_hallucinations / total)
print("Explanation & Error Locations Multiple errors: ", multiple_err_loc / total)
print("Explanation non-consistency error: ", explanation_consistency_err / total)
print("Repeat percentage: ", num_repeat / total_instances)
print("Total Instance: ", total_instances)
print("Total Errors: ", total)
