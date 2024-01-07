import glob
import json
from scipy import stats

gt_scores = open('test_caption_crowd/caption_crowd.score', 'r').readlines()
gt_scores = [float(ele[:-1].split('\t')[1]) for ele in gt_scores]

final_ls = []
for file_addr in sorted(glob.glob('test_caption_crowd/output_462/outputs/*')):
    lines = open(file_addr, 'r').readlines()
    final_ls+=lines

scores_ls = []
for ele in ''.join(final_ls).split('[SEP_WENDA]')[:-1]:
    score = 0
    score += (-1) * ele.count('Major/minor:  Minor')
    score += (-1) * ele.count('Major/minor: Minor')
    score += (-5) * ele.count('Major/minor:  Major')
    score += (-5) * ele.count('Major/minor: Major')
    if score <= -5:
        scores_ls+=[0]
    else:
        scores_ls+=[1]

res = stats.kendalltau(gt_scores, scores_ls)[0]
print("Kendall: ", res)
res = stats.spearmanr(gt_scores, scores_ls)[0]
print("Spearman: ", res)
res = stats.pearsonr(gt_scores, scores_ls)[0]
print("Pearson: ", res)
