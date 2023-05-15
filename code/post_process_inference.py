import json
from scipy import stats

# sys_ls = ['AISP-SJTU', 'HuaweiTSC', 'JDExploreAcademy', 'Lan-Bridge', 'LanguageX', 'M2M100_1.2B-B4', 'Online-A', 'Online-B', \
# 'Online-G', 'Online-W', 'Online-Y', 'bleu_bestmbr', 'bleurt_bestmbr', 'comet_bestmbr']
from mt_metrics_eval import data

wmt = "wmt22"
src_ref = "ref"
lang = "zh-en"
if wmt == "wmt22" and src_ref == "ref" and lang == "zh-en":
    suffix = ""
else:
    suffix = "_0_None"

evs = data.EvalSet(wmt, lang)
mqm_scores = evs.Scores("seg", "mqm")
sys_ls = set(mqm_scores) - evs.human_sys_names

new_sys_ls = []
for sys in sys_ls:
    if mqm_scores[sys][0] != None:
        new_sys_ls += [sys]

print(sys_ls)
print(evs.std_ref)

sescore3_ls, gt_ls = [], []
for sys in new_sys_ls:
    ref_data = json.load(
        open(
            f"test_{wmt}_{lang}/{src_ref}/test_{wmt}_{lang}_{sys}_llama_{src_ref}_data.json"
        )
    )
    ref_ls = [ele["output"] for ele in ref_data["instances"]]
    final_str = ""
    lines = open(
        f"test_{wmt}_{lang}/sescore3_{src_ref}_output/test_{wmt}_{lang}_{sys}_llama_{src_ref}_data{suffix}.txt",
        "r",
    ).readlines()
    for ele in lines:
        final_str += ele

    print("Test sample: ", len(final_str.split("[SEP_WENDA]")[:-1]))
    scores_ls = []
    for ele in final_str.split("[SEP_WENDA]")[:-1]:
        score = 0
        score += (-1) * ele.count("Major/minor: Minor")
        score += (-5) * ele.count("Major/minor: Major")
        scores_ls += [score]

    for sescore3, gt in zip(scores_ls, ref_ls):
        # if gt < -5:
        sescore3_ls += [sescore3]
        gt_ls += [gt]

res = stats.kendalltau(sescore3_ls, gt_ls)
print(res)
res = stats.pearsonr(sescore3_ls, gt_ls)
print(res)
