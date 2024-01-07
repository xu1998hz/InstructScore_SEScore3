import json
from scipy import stats

def count_loc_funct(text):
    if "Error location 5:" in text:
        return 5
    elif "Error location 4:" in text:
        return 4
    elif "Error location 3:" in text:
        return 3
    elif "Error location 2:" in text:
        return 2
    elif "Error location 1:" in text:
        return 1 
    else:
        print(text)
        print("Error!")
        exit(1)

from mt_metrics_eval import data

wmt = "wmt22"
src_ref = "ref"
lang = "zh-en"
suffix = "_0_None"

evs = data.EvalSet(wmt, lang)
mqm_scores = evs.Scores("seg", "mqm")
sys_ls = set(mqm_scores) - evs.human_sys_names # {evs.std_ref}

new_sys_ls = []
for sys in sys_ls:
    if mqm_scores[sys][0] != None:
        new_sys_ls += [sys]

sescore3_ls, gt_ls = [], []
count = 0
final_dict = {}
txt_final_dict = {}
for sys in new_sys_ls:
    final_dict[sys] = {}
    ref_data = json.load(
        open(
            f"test_{wmt}_{lang}/{src_ref}/test_{wmt}_{lang}_{sys}_llama_{src_ref}_data.json"
        )
    )
    ref_ls = [ele["output"] for ele in ref_data["instances"]]
    final_str = ""
    lines = open(
        f"test_{wmt}_{lang}/SEScore3_output_264/test_{wmt}_{sys}_llama_{src_ref}_data{suffix}.txt",
        "r",
    ).readlines()
    for ele in lines:
        final_str += ele

    scores_ls = []
    err_loc_ls = []
    cur_txt_ls = []
    for ele in final_str.split("[SEP_WENDA]")[:-1]:
        score = 0
        score += (-1) * ele.count("Major/minor: Minor")
        score += (-5) * ele.count("Major/minor: Major")
        scores_ls += [score]
        cur_txt_ls += [ele]
        
        if score != 0:
            temp_err_loc_ls = []
            for i in range(1, count_loc_funct(ele)+1):
                cur_txt = ele.split(f"Error location {i}:")[1].split('\n')[0]
                temp_err_loc_ls += [cur_txt.replace('"', '').strip()]
            err_loc_ls += [temp_err_loc_ls]
        else:
            err_loc_ls += ["no_error"]

    final_dict[sys]['err_locs'] = err_loc_ls
    final_dict[sys]['scores'] = scores_ls
    txt_final_dict[sys] = cur_txt_ls
    for sescore3, gt in zip(scores_ls, ref_ls):
        sescore3_ls += [sescore3]
        gt_ls += [gt]

res = stats.kendalltau(sescore3_ls, gt_ls)
print("Kendall: ", res[0])
res = stats.spearmanr(sescore3_ls, gt_ls)
print("Spearman: ", res[0])
res = stats.pearsonr(sescore3_ls, gt_ls)
print("Pearson: ", res[0])