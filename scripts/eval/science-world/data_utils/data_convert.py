import json
import os
import tqdm
from data_utils import clean, add_current_place, add_current_objects, compose_instance_v1, compose_instance_v2, compose_instance_v4, compose_instance_v5, downsampling, get_real_task_id
import argparse
from collections import defaultdict, Counter

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type = str, default = 'fast_system', help = 'mode of data')
parser.add_argument('--dir', type = str, default = 'data_v4', help = 'mode of data')
parser.add_argument('--lite', action = 'store_true')
parser.add_argument('--uniform', action = 'store_true')
parser.add_argument('--data_split', action = 'store_true', help = 'split subtasks into train/test subtask')

args = parser.parse_args()

# update variables based on arguments
mode = args.mode
# k_FiD = args.k_FiD
# FiD = args.FiD
# LongT5 = args.LongT5
data_split = args.data_split
# timestep = args.timestep
K = 10

# golds = glob('data/gold/*.json')
gold_data_path = "goldsequences-0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23-24-25-26-27-28-29.json"
# for gold in golds:
with open(gold_data_path, 'r') as f:
    raw_data = json.load(f) 

data = []
train_data = []
val_data = []
test_data = []

train_data_by_task = defaultdict(list)
val_data_by_task = defaultdict(list)
test_data_by_task = defaultdict(list)

 
training_task_ids = raw_data.keys()

print(training_task_ids)

all_actions = []
task_idx_real_distribution = []
task_id_to_vars = {}
task_id_to_actions = {}
# for raw_data in all_raw_data: 
for task_id in tqdm.tqdm(training_task_ids, desc="processing the data"):
    curr_task = raw_data[task_id]
    task_name = curr_task["taskName"]
    task_idx = curr_task["taskIdx"]
    if task_name.startswith("task"):  
        second_index = task_name.index('-', task_name.index('-') + 1)
        task_name = task_name[second_index+1:]
        task_name = task_name.replace("(","")
        task_name = task_name.replace(")","") 
    print(task_name)
    task_idx_real = get_real_task_id(task_name)
    task_group_id = task_idx_real.split("-")[0]
    curr_task_seq = curr_task['goldActionSequences']
     
    curr_task_seq = downsampling(task_idx_real, curr_task_seq)

    print(f"task_id: {task_id}; task_idx: {task_idx}; task_idx_real: {task_idx_real}")
    print(f"Task name: {task_name};  #Vars: {len(curr_task_seq)}")
    task_id_to_vars[task_id] = {"train":0, "dev": 0, "test": 0}
    task_id_to_actions[task_id] = {"train":0, "dev": 0, "test": 0}
    # Start data processing
    for seq_sample in curr_task_seq:
        task_desc = seq_sample['taskDescription']
        VarId = seq_sample['variationIdx'] 
        places = []
        objects = [] 

        original_steps = seq_sample['path']

        steps = []
        for s in original_steps:
            if s['action'] == "look around":
                continue
            if s["action"].startswith("close door"):
                continue
            steps.append(s)
 
        if len(steps) < 2:
            continue
        fold = seq_sample['fold']
        
        task_id_to_vars[task_id][fold] += 1
        
        obs = steps[0]['observation']
        action = steps[0]['action']

        gold_length = len(steps)
        # filtering steps 



        for i in range(1, len(steps)):  # i is from the 2nd step
            
            if i >= 2:
                # start = max(1, i-K)
                # for j in range(start, i): # no i-1
                # if steps[i-1]['observation'] != "The door is already open.":
                recent_actions.append(steps[i-1]['action'])
                recent_obs.append(steps[i-1]['observation'])
                recent_scores.append(float(steps[i-1]['score']))
                recent_reward.append(recent_scores[-1]-recent_scores[-2])
            else: 
                recent_actions = ["look around"]
                recent_obs = ["N/A"]
                recent_scores = [0]
                recent_reward = [0]

            prev_step = steps[i - 1]
            curr_step = steps[i]

            prev_prev_step = steps[i - 2] if i >= 2 else None

            returns_to_go = 1.0 - float(prev_step['score'])
            returns_to_go = round(returns_to_go, 2)

            prev_action = prev_step['action']
            curr_action = curr_step['action']
            prev_obs = prev_prev_step['observation'] if i >= 2 else "N/A"
            curr_obs = prev_step['observation']
            look = curr_step['freelook']
            prev_look = prev_step['freelook']
            inventory = curr_step['inventory']

            # Extract current place
            add_current_place(curr_obs, look, places)

            # Extract objects
            add_current_objects(task_id, look, objects, limit=25)

            # if curr_obs.find("move to the") != -1:
            #     add_current_objects(task_id, prev_look, objects, limit=20)

            """
            def compose_instance(mode, step_id, task_desc, returns_to_go, curr_action,
                     curr_obs, inventory, look, prev_action, prev_obs, 
                     objects, places, 
                     recent_actions)
            """

            input_str, label = compose_instance_v4(mode, i, task_desc, returns_to_go, curr_action,
                                    curr_obs, inventory, look, prev_action, prev_obs, 
                                    objects, places, 
                                    recent_actions, recent_obs, recent_scores, recent_reward) 

            curr_dat = {'input': input_str, 'target': label}
            
            curr_dat["task_id"] = int(task_id)
            curr_dat["variation_id"] = VarId
            curr_dat["task_real_id"] = task_idx_real
            task_idx_real_distribution.append(task_idx_real)
            all_actions.append(label)

            task_id_to_actions[task_id][fold] += 1
            
            if fold == 'train':
                train_data.append(curr_dat)
                train_data_by_task[task_group_id].append(curr_dat)
            elif fold == 'dev':
                val_data.append(curr_dat)
                val_data_by_task[task_group_id].append(curr_dat)
            elif fold == 'test':
                test_data.append(curr_dat)
                test_data_by_task[task_group_id].append(curr_dat)
 
 
## show stat

# print(json.dumps(task_id_to_vars, indent=2))
# print(json.dumps(task_id_to_actions, indent=2))

for tid, v_stat in task_id_to_vars.items():
    print(f'{tid}, {v_stat["train"]}, {v_stat["dev"]}, {v_stat["test"]}')

for tid, v_stat in task_id_to_actions.items():
    print(f'{tid}, {v_stat["train"]}, {v_stat["dev"]}, {v_stat["test"]}')


counter = Counter(task_idx_real_distribution)
for value, count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
    print(f"{value}: {count}")

action_counter = Counter([a.split()[0] for a in all_actions])
for value, count in sorted(action_counter.items(), key=lambda x: x[1], reverse=True):
    print(f"{value}: {count}")

with open(f"{args.dir}/{mode}.train.jsonl", 'w') as f:
    for item in train_data:
        f.write(json.dumps(item) + "\n")

with open(f"{args.dir}/{mode}.val.jsonl", 'w') as f:
    for item in val_data:
        f.write(json.dumps(item) + "\n")

with open(f"{args.dir}/{mode}.val.mini.jsonl", 'w') as f:
    import random
    random.seed(1)
    random.shuffle(val_data)
    val_data = val_data[:10000]
    for item in val_data:
        f.write(json.dumps(item) + "\n")

with open(f"{args.dir}/{mode}.test.jsonl", 'w') as f:
    for item in test_data:
        f.write(json.dumps(item) + "\n") 


task_real_ids = list(train_data_by_task.keys())
assert len(train_data_by_task) == len(val_data_by_task) == len(test_data_by_task)
for trid in task_real_ids:
    all_data = {"train": train_data_by_task, 
                "val": val_data_by_task, 
                "test": test_data_by_task}
    for split in all_data:
        data = all_data[split][trid]
        with open(f"{args.dir}/data_dir/{mode}.{trid}.{split}.json", 'w') as f:
            for item in data:
                f.write(json.dumps(item) + "\n") 
        if split == "val":
            with open(f"{args.dir}/data_dir/{mode}.{trid}.{split}.mini.json", 'w') as f:
                for item in data[:2000]:
                    f.write(json.dumps(item) + "\n") 