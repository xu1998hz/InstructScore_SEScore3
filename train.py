from peft import get_peft_model, LoraConfig, TaskType
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import click
import torch
import wandb
import random
from torch.utils.data import DataLoader
from datasets import Dataset
import os
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    AdamW
)
import math
import time
import json

# needs to adjust corresponding FLAN-T5
class exp_config():
    max_length = 1024
    lr = 0.0001
    gradient_accumulation_steps=6
    temp = 0.1
    drop_out=0.1
    activation="Tanh"
    final_activation=None
    vali_size=200

# process the data so that input and output data can be segmented
def prepare_data(inp_lines, out_lines):
    in_str = ''
    for line in inp_lines:
        in_str+=line
    out_str = ''
    for line in out_lines:
        out_str+=line
    in_ls, out_ls = in_str.split('[SEP]')[:-1], out_str.split('[SEP]')[:-1]
    in_ls = [ele[:-1] for ele in in_ls]
    out_ls = [ele[:-1] for ele in out_ls]
    return in_ls, out_ls 

def preprocess_data(in_ls, out_ls, tokenizer, max_length, batch_size, mode="train", shuffle=True, sampler=True):
    if mode=="train":
        ds = Dataset.from_dict({"input": in_ls, 'output': out_ls})
    else:
        ds = Dataset.from_dict({"input": in_ls})

    def preprocess_function(examples):
        model_data = {}
        # model inputs
        inputs = tokenizer(examples['input'], max_length=max_length, padding='max_length', truncation=True)
        model_data['input_ids'], model_data['attn_masks'] = inputs['input_ids'], inputs['attention_mask']
        # model ground truth outputs
        if mode == "train":
            outputs = tokenizer(examples['output'], max_length=max_length, padding='max_length', truncation=True)
            model_data['output_ids'] = outputs['input_ids']
        return model_data

    processed_datasets = ds.map(
        preprocess_function,
        batched=True,
        num_proc=None,
        remove_columns=ds.column_names,
        desc="Running tokenizer on dataset",
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding='max_length',
        max_length=max_length,
        pad_to_multiple_of=None,
        return_tensors = 'pt'
    )

    if sampler:
        data_sampler = torch.utils.data.distributed.DistributedSampler(processed_datasets, shuffle=shuffle)
        dataloader = DataLoader(processed_datasets, batch_size=batch_size, collate_fn=data_collator, sampler=data_sampler)
    else:
        dataloader = DataLoader(processed_datasets, batch_size=batch_size, collate_fn=data_collator, shuffle=shuffle)
    return dataloader

@click.command()
@click.option('-flan_type')
@click.option('-peft', default=None)
@click.option('-inf')
@click.option('-outf')
@click.option('-num_epoch', default=1, type=int)
@click.option('-batch_size', default=6, type=int)
@click.option('-save_dir_name', default='flanT5_weights', type=str)
@click.option('-eval_step', type=int)
@click.option('-inference', type=bool, default=False)
def main(flan_type, peft, inf, outf, num_epoch, batch_size, save_dir_name, eval_step, inference):
    model_name_or_path = f"google/flan-t5-{flan_type}"
    tokenizer_name_or_path = f"google/flan-t5-{flan_type}"

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    # enable parameter efficient weight updates
    if peft == 'lora':
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)

    # initalize the process
    dist.init_process_group(backend="nccl")
    rank = int(os.environ['LOCAL_RANK'])
    # only main process initalize wandb
    if rank == 0:
        # initalize the project parameters into Wandb, store experiment specific parameters
        wandb.init(project="COTScore", config=
        {   
            "model size": flan_type,
            "strategy": "Seq-to-Seq",
            "epoch": num_epoch,
            "eval_step": eval_step,
            "train batch size": batch_size * exp_config.gradient_accumulation_steps * 8,
            "lr": exp_config.lr,
        })

    exp_config.device_id = rank % torch.cuda.device_count()
    # set cuda device with rank and clear ram cache to ensure balanced ram allocations
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    model.to(exp_config.device_id)
    # parallelize the pipeline into multiple gpus
    optimizer = AdamW(model.parameters(), lr=exp_config.lr)
    # loaded data in segments
    data = json.load(open('filter_ref_based_data_april_6.json'))
    train_in_ls, train_out_ls = [], []
    for _, ele in data.items():
        train_in_ls+=[ele['input']]
        train_out_ls+=[ele['output']]

    # vali_in_ls, vali_out_ls = in_ls[:exp_config.vali_size], out_ls[:exp_config.vali_size]
    # print("Vali Inputs: ", len(vali_in_ls))
    # print("Vali Outputs: ", len(vali_out_ls))
    # train_in_ls, train_out_ls = in_ls[exp_config.vali_size:], out_ls[exp_config.vali_size:]
    # print("Train Inputs: ", len(train_in_ls))
    # print("Train Outputs: ", len(train_out_ls))
    # print(train_in_ls[1])
    # print(train_out_ls[1])

    # needs sample and shuffle for training
    train_dataloader = preprocess_data(train_in_ls, train_out_ls, tokenizer, exp_config.max_length, batch_size, mode="train", shuffle=True, sampler=True)
    # vali_dataloader = preprocess_data(vali_in_ls, vali_out_ls, tokenizer, exp_config.max_length, batch_size, mode="train", shuffle=False, sampler=False)
    # if inference:
    #     # load in testing dataset
    #     pre_test_in_ls, pre_test_out_ls = open(ein, 'r').readlines(), open(eout, 'r').readlines()
    #     test_in_ls, test_out_ls = prepare_data(pre_test_in_ls, pre_test_out_ls)
    #     test_dataloader = preprocess_data(test_in_ls, test_out_ls, tokenizer, exp_config.max_length, batch_size, mode="test", shuffle=False, sampler=False)
    
    model = DDP(model, device_ids=[exp_config.device_id], find_unused_parameters=True)
    model.train()

    max_train_steps = math.ceil(len(train_dataloader) / exp_config.gradient_accumulation_steps) * num_epoch
    num_warmup_steps=0
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # save at end of epoch and at main processls
    if not os.path.isdir(f'{save_dir_name}') and rank == 0:
        os.makedirs(f'{save_dir_name}')
        print(f'{save_dir_name} is created!')

    global_step = 0 

    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(num_epoch):
            # reset the best correlation each time at start of epoch
            # cur_best_vali = float('inf')
            torch.cuda.empty_cache() # empty cache in gpus
            train_dataloader.sampler.set_epoch(epoch) # set the sampler at each epoch
            for step, train_batch in enumerate(train_dataloader):
                # accumulate losses at weights, each is normalized by accumulation steps
                train_outputs = model(input_ids=train_batch['input_ids'], attention_mask=train_batch['attn_masks'], labels=train_batch['output_ids'])
                train_loss = torch.mean(train_outputs.loss) / exp_config.gradient_accumulation_steps

                # evaluate at every eval_step and also at the end of epoch (includes the beginning loss)
                if ((step % (eval_step * exp_config.gradient_accumulation_steps) == 0) or (step == len(train_dataloader) - 1)) and rank == 0:
                    print("start to evaluate!")
                     # store all the losses in wandb
                    wandb_temp_dict = {}
                    # evaluate on the validation loss 
                    # start_eval_time = time.time()
                    # model.eval()

                    # vali_loss, count = 0, 0, 
                    # # vali_txt_ls = [] 
                    # # evaluate validation loss here
                    # with torch.no_grad():
                    #     for vali_batch in vali_dataloader:
                    #         vali_outputs=model(input_ids=vali_batch['input_ids'], attention_mask=vali_batch['attn_masks'], labels=vali_batch['output_ids'])
                    #         vali_loss+=torch.mean(vali_outputs.loss)
                    #         count+=1 
                    #         # vali_txt_ls += hypotheses_batch
                    #     vali_gen = model.module.generate(input_ids=vali_batch['input_ids'].to(exp_config.device_id), \
                    #                     attention_mask=vali_batch['attn_masks'].to(exp_config.device_id), max_new_tokens=exp_config.max_length)
                    #     hypotheses_batch = tokenizer.batch_decode(vali_gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    #     print(hypotheses_batch)
                    #     vali_loss /= count 

                    #vali_txt_ls = [vali+'[SEP]' for vali in vali_txt_ls]

                    print("Train Loss: ", train_loss.item())
                    # print("Vali Loss: ", vali_loss.item())
                    # wandb_temp_dict.update({"Vali Loss": vali_loss.item()})
                    wandb_temp_dict.update({"Train Loss": train_loss.item()})

                    # save at the best epoch step
                    if step == len(train_dataloader) - 1:
                        # cur_best_vali=vali_loss
                        # saveValiFile = open(f'{save_dir_name}/epoch{epoch}_step{step}.txt', 'w')
                        # saveValiFile.writelines(vali_txt_ls)
                        # print("File is saved!")
                        torch.save(model.module, f'{save_dir_name}/epoch{epoch}_end.ckpt')
                        print("Weight is saved!")

                    model.train()
                    torch.cuda.empty_cache()
                    wandb.log(wandb_temp_dict)
                    # print("Testing Duration: ", time.time()-start_eval_time)
                    # print('--------------------------------------------------------------')
                
                train_loss.backward()

                if step % exp_config.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad() # clear the grads

                global_step += 1

if __name__ == "__main__":
    random.seed(10)
    main()


