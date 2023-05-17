import json
from scipy import stats
import glob

task_name = 'webnlg'

gt_ls = []
sescore3_ls = []
for sys_addr in glob.glob('test_webnlg/ref/*'):
    gt_dict = json.load(open(sys_addr))
    gt_ls += [ele['output'] for ele in gt_dict['instances']]

    final_str = ''
    file_name = sys_addr.split('/')[-1].replace('.json', '_0_None.txt')
    lines = open(f'test_{task_name}/output_222/{file_name}', 'r').readlines()
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

res = stats.kendalltau(gt_ls, sescore3_ls)
print(res)