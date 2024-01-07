import json
from scipy import stats
import glob

task_name = 'webnlg'

gt_ls = []
sescore3_ls = []
cor_ls = []
cov_ls = []
flu_ls = []
rel_ls = []
str_ls = []
for sys_addr in glob.glob('test_webnlg/ref/*'):
    gt_dict = json.load(open(sys_addr))
    gt_ls += [sum(ele['output'].values()) for ele in gt_dict['instances']]
    cor_ls += [ele['output']['Correctness'] for ele in gt_dict['instances']]
    cov_ls += [ele['output']['DataCoverage'] for ele in gt_dict['instances']]
    flu_ls += [ele['output']['Fluency'] for ele in gt_dict['instances']]
    rel_ls += [ele['output']['Relevance'] for ele in gt_dict['instances']]
    str_ls += [ele['output']['TextStructure'] for ele in gt_dict['instances']]

    final_str = ''
    file_name = sys_addr.split('/')[-1].replace('.json', '_0_None.txt')
    lines = open(f'test_{task_name}/output_800/{file_name}', 'r').readlines()
    for ele in lines:
        final_str += ele

    scores_ls = []
    for ele in final_str.split('[SEP_WENDA]')[:-1]:
        score = 0
        score += (-1) * ele.count('Major/minor:  Minor')
        score += (-1) * ele.count('Major/minor: Minor')
        score += (-1) * ele.count('Major/minor:  Major')
        score += (-1) * ele.count('Major/minor: Major')
        scores_ls += [score]
    sescore3_ls += scores_ls

res = stats.kendalltau(gt_ls, sescore3_ls)
print("Kendall: ", res[0])
res = stats.pearsonr(gt_ls, sescore3_ls)
print("Pearson: ", res[0])

res = stats.kendalltau(cor_ls, sescore3_ls)
print("Correctness Kendall: ", res[0])

res = stats.kendalltau(cov_ls, sescore3_ls)
print("Coverage Kendall: ", res[0])

res = stats.kendalltau(flu_ls, sescore3_ls)
print("Fluency Kendall: ", res[0])

res = stats.kendalltau(rel_ls, sescore3_ls)
print("Relvence Kendall: ", res[0])

res = stats.kendalltau(str_ls, sescore3_ls)
print("TextStructure Kendall: ", res[0])