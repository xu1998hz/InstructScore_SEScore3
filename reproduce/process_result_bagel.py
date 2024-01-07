import glob
from scipy import stats

test_bagel_ls = []
test_bagel_ls_1, test_bagel_ls_2, test_bagel_ls_3 = [], [], []
index_ls = []

for i in range(8):
    file_addr = f"test_bagel/ref/bagel_gt_{i}.txt"
    lines = open(file_addr, "r").readlines()
    for line in lines:
        index = int(line[:-1].split("\t")[0])
        score_1, score_2, score_3 = (
            float(line[:-1].split("\t")[1]),
            float(line[:-1].split("\t")[2]),
            float(line[:-1].split("\t")[3]),
        )
        index_ls += [index]
        test_bagel_ls_1 += [score_1]
        test_bagel_ls_2 += [score_2]
        test_bagel_ls_3 += [score_3]
        test_bagel_ls += [(score_1 + score_2 + score_3) / 3]

final_score_ls = []
for i in range(8):
    lines = open(
        f"test_bagel/output_224/test_bagel_{i}_llama_ref_data_0_None.txt", "r"
    ).readlines()
    cur_str = "".join(lines)
    for ele in cur_str.split("[SEP_WENDA]")[:-1]:
        score = 0
        score += (-1) * ele.count("Major/minor: Minor")
        score += (-1) * ele.count("Major/minor:  Minor")
        score += (-5) * ele.count("Major/minor: Major")
        score += (-5) * ele.count("Major/minor:  Major")
        final_score_ls += [score]

index_gt_score_dict = {}
index_gt_score_dict_1 = {}
index_gt_score_dict_2 = {}
index_gt_score_dict_3 = {}
index_our_score_dict = {}
for index, gt_score, our_score in zip(index_ls, test_bagel_ls, final_score_ls):
    index_gt_score_dict[index] = gt_score
    if index not in index_our_score_dict:
        index_our_score_dict[index] = []
    index_our_score_dict[index] += [our_score]

for index, gt_score in zip(index_ls, test_bagel_ls_1):
    index_gt_score_dict_1[index] = gt_score

for index, gt_score in zip(index_ls, test_bagel_ls_2):
    index_gt_score_dict_2[index] = gt_score

for index, gt_score in zip(index_ls, test_bagel_ls_3):
    index_gt_score_dict_3[index] = gt_score

for index in index_our_score_dict:
    index_our_score_dict[index] = max(index_our_score_dict[index])

res = stats.kendalltau(
    list(index_gt_score_dict.values()), list(index_our_score_dict.values())
)
print("Kendall Total: ", res[0])

res = stats.pearsonr(
    list(index_gt_score_dict.values()), list(index_our_score_dict.values())
)
print("Pearson Total: ", res[0])

res = stats.kendalltau(
    list(index_gt_score_dict_1.values()), list(index_our_score_dict.values())
)
print("1: ", res)
res = stats.kendalltau(
    list(index_gt_score_dict_2.values()), list(index_our_score_dict.values())
)
print("2: ", res)
res = stats.kendalltau(
    list(index_gt_score_dict_3.values()), list(index_our_score_dict.values())
)
print("3: ", res)
