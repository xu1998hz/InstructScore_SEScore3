from mt_metrics_eval import data
import random
import json

lang = 'zh-en'
evs = data.EvalSet('wmt20', 'zh-en')

save_train_dict, save_test_dict = {"type": "text2text", "instances": []}, {"type": "text2text", "instances": []}
ref_ls = evs.sys_outputs[evs.std_ref]
for i in range(2000):
    cur_set = random.sample(list(set(evs.sys_outputs)-{evs.std_ref}), 10)
    reward_set, update_set = cur_set[:5], cur_set[5:]
    for sys in reward_set:
        cur_output = evs.sys_outputs[sys][i]
        cur_ref = ref_ls[i]
        train_prompt = f"You are evaluating Chinese-to-English Machine translation task. The correct translation is \"{cur_ref}\". The model generated translation is \"{cur_output}\". Please identify all errors within each model output, up to a maximum of five. For each error, please give me the corresponding error type, major/minor label, error location of the model generated translation and explanation for the error. Major errors can confuse or mislead the reader due to significant change in meaning, while minor errors don't lead to loss of meaning but will be noticed."
        save_train_dict["instances"] += [{"input": train_prompt}]
    for sys in update_set:
        test_prompt = f"You are evaluating Chinese-to-English Machine translation task. The correct translation is \"{cur_ref}\". The model generated translation is \"{cur_output}\". Please identify all errors within each model output, up to a maximum of five. For each error, please give me the corresponding error type, major/minor label, error location of the model generated translation and explanation for the error. Major errors can confuse or mislead the reader due to significant change in meaning, while minor errors don't lead to loss of meaning but will be noticed."
        save_test_dict["instances"] += [{"input": test_prompt}]

with open(f'data/english_llama_ref_reward.json', 'w') as f:
    json.dump(save_train_dict, f)

with open(f'data/english_llama_ref_test.json', 'w') as f:
    json.dump(save_test_dict, f)

print("Reward and test files are saved!")