import json
from scipy import stats

# sys_ls = ['AISP-SJTU', 'HuaweiTSC', 'JDExploreAcademy', 'Lan-Bridge', 'LanguageX', 'M2M100_1.2B-B4', 'Online-A', 'Online-B', \
# 'Online-G', 'Online-W', 'Online-Y', 'bleu_bestmbr', 'bleurt_bestmbr', 'comet_bestmbr']
from mt_metrics_eval import data

wmt='wmt22'
evs = data.EvalSet(wmt, 'zh-en')
mqm_scores = evs.Scores('seg', 'mqm')
sys_ls = set(mqm_scores) - evs.human_sys_names  #-{evs.std_ref}
print(evs.std_ref)
print(sys_ls)
sescore3_ls, gt_ls = [], []
for sys in sys_ls:
    ref_data = json.load(open(f'test_{wmt}_zh-en/ref/test_{wmt}_zh-en_{sys}_llama_ref_data.json'))
    ref_ls = [ele['output'] for ele in ref_data['instances']]
    final_str = ''
    lines = open(f'test_{wmt}_zh-en/SEScore3_output_222/test_{wmt}_zh-en_{sys}_llama_ref_data_0_None.txt', 'r').readlines()
    for ele in lines:
        final_str += ele

    print("Test sample: ", len(final_str.split('[SEP_WENDA]')[:-1]))
    scores_ls = []
    for ele in final_str.split('[SEP_WENDA]')[:-1]:
        score = 0
        score += (-1) * ele.count('Major/minor: Minor')
        score += (-5) * ele.count('Major/minor: Major')
        scores_ls += [score]
    sescore3_ls += scores_ls
    gt_ls += ref_ls

res = stats.kendalltau(sescore3_ls, gt_ls)
print(res)
res = stats.pearsonr(sescore3_ls, gt_ls)
print(res)