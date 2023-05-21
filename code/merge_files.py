import glob

final_dict = {}
for file_addr in glob.glob(
    "/share/edc/home/wendaxu/reward_selections/SEScore3_output_sample_222/*_3.txt"
):
    file_name = file_addr.split("/")[-1]
    sys_name = file_name.split("test_wmt22_zh-en_")[1].split("_llama_")[0]
    final_str = ""
    lines = open(file_addr, "r").readlines()
    for ele in lines:
        final_str += ele
    if sys_name not in final_dict:
        final_dict[sys_name] = {}
    for cur_str in final_str.split("[SEP_WENDA]")[:-1]:
        sen_index, ret_index, output = (
            cur_str.split("\t")[0],
            cur_str.split("\t")[1],
            cur_str.split("\t")[2],
        )
        if sen_index not in final_dict[sys_name]:
            final_dict[sys_name][sen_index] = {}
        final_dict[sys_name][sen_index][ret_index] = output

sys_ls = [
    "LanguageX",
    "M2M100_1.2B-B4",
    "Online-A",
    "Online-B",
    "Online-G",
    "Online-W",
    "Online-Y",
    "bleurt_bestmbr",
]

output_ls = []
for sys in sys_ls:
    print(final_dict[sys].keys())
    print()
    for ele in list(final_dict[sys].values()):
        output_ls += [cur_ele for cur_ele in ele.values()]
