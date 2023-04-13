import json
from scipy import stats

sys_ls = ['AISP-SJTU', 'HuaweiTSC', 'JDExploreAcademy', 'Lan-Bridge', 'LanguageX', 'M2M100_1.2B-B4', 'Online-A', 'Online-B', \
'Online-G', 'Online-W', 'Online-Y', 'bleu_bestmbr', 'bleurt_bestmbr', 'comet_bestmbr']
sescore3_ls, gt_ls = [], []
for sys in sys_ls:
    ref_data = json.load(open(f'test_wmt22_zh-en/ref/test_wmt22_zh-en_{sys}_llama_ref_data.json'))
    ref_ls = [ele['output'] for ele in ref_data['instances']]
    final_str = ''
    lines = open(f'test_wmt22_zh-en/output_222/test_wmt22_zh-en_{sys}_llama_ref_data.txt', 'r').readlines()
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