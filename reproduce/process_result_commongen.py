import glob
import math
from scipy import stats


def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i : min(i + batch_size, len(lst))]


final_str = ""
for file_addr in sorted(glob.glob("test_commongen/output_462/*")):
    lines = open(file_addr, "r").readlines()
    final_str += "".join(lines)

lines = open("test_commongen/common_gen_test_1.txt", "r").readlines()
lines = [float(ele[:-1].split("\t")[-1]) for ele in lines]


def score_funct(ele):
    score = 0
    score += (-1) * ele.count("Major/minor:  Minor")
    score += (-5) * ele.count("Major/minor:  Major")
    return score


p, q = 0, 0
n0, n1, n2 = 0, 0, 0
h_ls, s_ls = [], []
for ele, hum in zip(
    batchify(final_str.split("[SEP_WENDA]")[:-1], 2), batchify(lines, 2)
):
    h_ls += [hum[0]]
    s_ls += [score_funct(ele[0]) - score_funct(ele[1])]
    if (hum[0] > 0 and score_funct(ele[0]) > score_funct(ele[1])) or (
        hum[0] < 0 and score_funct(ele[0]) < score_funct(ele[1])
    ):
        p += 1
    elif (
        hum[0] < 0
        and score_funct(ele[0]) > score_funct(ele[1])
        or (hum[0] > 0 and score_funct(ele[0]) < score_funct(ele[1]))
    ):
        q += 1
    elif hum[0] == 0 and score_funct(ele[0]) == score_funct(
        ele[1]
    ):  # and score_funct(ele[1]) == score_funct(ele[0])
        n0 += 1
    elif hum[0] == 0 and score_funct(ele[0]) != score_funct(ele[1]):
        n1 += 1
    elif hum[0] != 0 and score_funct(ele[0]) == score_funct(ele[1]):
        n2 += 1
    else:
        print("nothing should be here!")

print("p, q, n0, n1, n2: ", p, q, n0, n1, n2)
print("InstructScore: ", (p + n0) / (p + q + n0 + n1 + n2))
