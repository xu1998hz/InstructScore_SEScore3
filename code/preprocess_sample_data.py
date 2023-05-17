import glob

# load in all sample data
cur_dict = {}
for file_name in glob.glob('data/wmt_samples/*'):
    lines = open(file_name, 'r').readlines()
    final_str = ''.join(lines)
    samples_ls = final_str.split('[SEP_WENDA]')[:-1]   
    for ele in samples_ls:
        if ele.split('\t')[0] not in cur_dict:
            cur_dict[ele.split('\t')[0]]=[]
        cur_dict[ele.split('\t')[0]] += [ele.split('\t')[1]]

new_dict = {}
for i in range(0, len(cur_dict), 5):
    print(cur_dict['0'][0])
    print()
    print(cur_dict['5'][0])