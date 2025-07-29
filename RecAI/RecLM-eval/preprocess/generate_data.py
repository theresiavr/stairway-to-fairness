# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Edited by theresiavr, mainly to load custom data splits and to load various prompt templates

import os
import re
import json
import gzip
import math
import random
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from multiprocessing import Pool
from collections import defaultdict

random.seed(43)

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def ReadLineFromFile(path):
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

def parse(path):
    if path.endswith('gz'):
        g = gzip.open(path, 'r')
    else:
        g = open(path, 'r')
    nan_default = {'NaN': "", 'false': "", 'true': ""}
    for l in g:
        yield eval(l, nan_default)

class Test_Dataset():
    def __init__(self, all_task_templates, dataset='steam', prompt_type="neutral"):
        self.all_task_templates = all_task_templates
        self.dataset = dataset  # dataset to use
        self.prompt_type = prompt_type
        # self.split = split  # train/valid/test

        if dataset=="steam":
            self.sequential_data = ReadLineFromFile(os.path.join('./data', dataset, 'sequential_data.txt'))
        else:
            self.sequential_data = ReadLineFromFile(os.path.join('./data', dataset, 'train.txt'))
        
        self. val_items = None
        if os.path.exists(os.path.join('./data', dataset, 'val.txt')):
            self.val_items = ReadLineFromFile(os.path.join('./data', dataset, 'val.txt'))

        self.test_items = None
        if os.path.exists(os.path.join('./data', dataset, 'test.txt')):
            self.test_items = ReadLineFromFile(os.path.join('./data', dataset, 'test.txt'))
        
        self.meta_data = ['padding']  # item_id start from 0

        self.raw_id2meta_id = {}
        for meta in parse(os.path.join('./data', dataset, 'metadata.json')):
            self.meta_data.append(meta)
            # meta["app_name"] = re.sub('[^A-Za-z0-9_.,!?;:\n ]', '', meta["app_name"])
            try:
                self.raw_id2meta_id[meta['item_id']] = len(self.meta_data) - 1
            except:
                self.raw_id2meta_id[meta['id']] = len(self.meta_data) - 1

        self.user_data = None

        if prompt_type == "sensitive":
            self.user_data = {}
            for user_data in parse(os.path.join('./data', dataset, 'user_data.json')):

                dict_this_user = {}
                for k,v in user_data.items():
                    if k == "user_id":
                        user_id = v
                    else:
                        dict_this_user[k] = v
                
                self.user_data[user_id] = dict_this_user

        self.repeat_text = None
        if os.path.exists(os.path.join('./data', dataset, 'repeat.txt')):
            self.repeat_text = ReadLineFromFile(os.path.join('./data', dataset, 'repeat.txt'))

        # if os.path.exists(os.path.join('./data', dataset, 'negative_samples.txt')): #may not need negative_samples
        #     self.negative_samples = ReadLineFromFile(os.path.join('./data', dataset, 'negative_samples.txt')) 

        # if os.path.exists(os.path.join('./data', dataset, 'item_datasets.pkl')):
            # self.test_items = load_pickle(os.path.join('./data', dataset, 'item_datasets.pkl'))['test']

        # self.search_data = None
        # if os.path.exists(os.path.join('./data', dataset, 'search_data.csv')):
        #     self.search_data = pd.read_csv(os.path.join('./data', dataset, 'search_data.csv'))

    def gen_retrieval_data(self, sample_num):
        datasets = []
        sample_num = min(sample_num, len(self.sequential_data))
        for idx in range(sample_num):
            retrieval_datum = self.sequential_data[idx]
            sequence = [int(x) for x in retrieval_datum.split()]

            #include user_id
            user_id = sequence[0]
            click_history = sequence[1:]

            history_titles = []

            for item_id in click_history:
                item_datum = self.meta_data[self.raw_id2meta_id[item_id]]
                item_title = 'unknown title'
                if 'app_name' in item_datum:
                    item_title = item_datum['app_name']
                elif 'title' in item_datum:
                    item_title = item_datum['title']
                history_titles.append(item_title)

            if self.repeat_text is not None:
                repeat_data = self.repeat_text[idx]
                rep_sequence = repeat_data.split()

                rep_user_id = int(rep_sequence[0])
                assert rep_user_id == user_id, "User ID from repeat.txt doesn't match train user ID"

                repeat_history = rep_sequence[1:]


            if self.val_items is not None:
                val_data = self.val_items[idx]
                val_sequence = [int(x) for x in val_data.split()]

                val_user_id = val_sequence[0]
                assert val_user_id == user_id, "Val user ID doesn't match train user ID"

                val_click_history = val_sequence[1:]
                
                if val_click_history[0] == -1:
                    user_has_val = False

                else:
                    user_has_val = True
                    val_titles = []

                    for item_id in val_click_history:

                        item_datum = self.meta_data[self.raw_id2meta_id[item_id]]
                        item_title = 'unknown title'
                        if 'title' in item_datum:
                            item_title = item_datum['title']
                        val_titles.append(item_title)

            
            if self.test_items is not None:
                test_data = self.test_items[idx]
                test_sequence = [int(x) for x in test_data.split()]

                test_user_id = test_sequence[0]
                assert test_user_id == user_id, "Test user ID doesn't match train user ID"

                test_click_history = test_sequence[1:]

                test_titles = []

                for item_id in test_click_history:
                    item_datum = self.meta_data[self.raw_id2meta_id[item_id]]
                    item_title = 'unknown title'
                    if 'title' in item_datum:
                        item_title = item_datum['title']
                    test_titles.append(item_title)


            else: #steam/default case
                target_item = sequence[-1]
                target_item_datum = self.meta_data[target_item]
                test_titles = 'unknown title'
                if 'app_name' in target_item_datum:
                    test_titles = target_item_datum['app_name']

            if self.user_data is not None:
                attr_for_user = self.user_data[user_id]

                if self.dataset == "ml-1m":
                    attr1 = attr_for_user["gender"]
                    attr2 = attr_for_user["age"]
                    attr3 = attr_for_user["occupation"]

                elif self.dataset == "lfm-1b":
                    attr1 = attr_for_user["gender"]
                    attr2 = attr_for_user["age"]
                    attr3 = attr_for_user["country_name"]
                    
                elif self.dataset == "jobrec":
                    attr1 = attr_for_user["DegreeType"]
                    attr3 = attr_for_user["TotalYearsExperience"]

                    if attr1 == "High School":
                        attr2 = " and"
                    else:
                        major = attr_for_user["Major"]
                        attr2 = f", majoring in {major},"


            task_template = self.all_task_templates['retrieval']


            if self.dataset in ["ml-1m", "jobrec"]:
                target_text = task_template['target'].format(', '.join(test_titles))

                if self.prompt_type == "neutral":

                    if user_has_val:
                        source_text = task_template['source'].format(', '.join(history_titles), ', '.join(val_titles))
                    else:
                        source_text = task_template['source_no_val'].format(', '.join(history_titles))
                        val_titles = []
                       
                elif self.prompt_type == "sensitive":
                    if user_has_val:
                        source_text = task_template['source'].format(attr1, attr2, attr3,', '.join(history_titles), ', '.join(val_titles))
                    else:
                        source_text = task_template['source_no_val'].format(attr1, attr2, attr3,', '.join(history_titles))

                datasets.append({
                    "id": user_id,
                    "source": source_text,
                    "target": target_text,
                    "val": val_titles,
                    "history": history_titles,
                    "task": "retrieval"
                })
            elif self.dataset == "lfm-1b":
                target_text = task_template['target'].format(', '.join(test_titles))

                if self.prompt_type == "neutral":
                    if user_has_val:
                        source_text = task_template['source'].format(', '.join(history_titles), ', '.join(repeat_history), ', '.join(val_titles))
                    else:
                        source_text = task_template['source_no_val'].format(', '.join(history_titles), ', '.join(repeat_history))
                        val_titles = []

                elif self.prompt_type == "sensitive":
                    if user_has_val:
                        source_text = task_template['source'].format(attr1, attr2, attr3,', '.join(history_titles), ', '.join(repeat_history), ', '.join(val_titles))
                    else:
                        source_text = task_template['source_no_val'].format(attr1, attr2, attr3,', '.join(history_titles), ', '.join(repeat_history))
                        val_titles = []

                datasets.append({
                    "id": user_id,
                    "source": source_text,
                    "target": target_text,
                    "val": val_titles,
                    "history": history_titles,
                    "task": "retrieval"
                })

            else:
                source_text = task_template['source'].format(', '.join(history_titles))
                target_text = task_template['target'].format(', '.join(test_titles))
                datasets.append({
                    "id": user_id,
                    "source": source_text,
                    "target": target_text,
                    "history": history_titles,
                    "task": "retrieval"
                })

            # source_text = re.sub('[^A-Za-z0-9_.,!?;:\n ]', '', source_text)
            # target_text = re.sub('[^A-Za-z0-9_.,!?;:\n ]', '', target_text)
        return datasets

    def gen_ranking_data(self, sample_num):
        datasets = []
        sample_num = min(sample_num, len(self.sequential_data))
        for idx in range(sample_num):
            ranking_datum = self.sequential_data[idx]
            sequence = [int(x) for x in ranking_datum.split()]
            click_history = sequence[1:-1]
            target_item = sequence[-1]
            history_titles = []
            for item_id in click_history:
                item_datum = self.meta_data[item_id]
                item_title = 'unknown title'
                if 'app_name' in item_datum:
                    item_title = item_datum['app_name']
                history_titles.append(item_title)
            
            target_item_datum = self.meta_data[target_item]
            target_item_title = 'unknown title'
            if 'app_name' in target_item_datum:
                target_item_title = target_item_datum['app_name']

            user_id = sequence[0]
            assert user_id == int(self.negative_samples[int(user_id)-1].split(' ', 1)[0])
            candidate_samples = self.negative_samples[int(user_id)-1].split(' ', 1)[1].split(' ')
            candidate_samples = random.sample(candidate_samples, 20)
            candidate_samples.extend([target_item])
            random.shuffle(candidate_samples)

            candidate_titles = []
            for item_id in candidate_samples:
                item_datum = self.meta_data[int(item_id)]
                item_title = 'unknown title'
                if 'app_name' in item_datum:
                    item_title = item_datum['app_name']
                candidate_titles.append(item_title)

            task_template = self.all_task_templates['ranking']
            source_text = task_template['source'].format(', '.join(history_titles), ', '.join(candidate_titles))
            target_text = task_template['target'].format(target_item_title)
            source_text = re.sub('[^A-Za-z0-9_.,!?;:\n ]', '', source_text)
            target_text = re.sub('[^A-Za-z0-9_.,!?;:\n ]', '', target_text)
            datasets.append({
                "source": source_text,
                "target": target_text,
                "history": history_titles,
                "candidate": candidate_titles,
                "task": "ranking"
            })
        return datasets

    def gen_explanation_data(self, sample_num):
        datasets = []
        sample_num = min(sample_num, len(self.sequential_data))
        for idx in range(sample_num):
            exp_datum = self.sequential_data[idx]
            sequence = [int(x) for x in exp_datum.split()]
            click_history = sequence[1:-1]
            target_item = sequence[-1]
            history_titles = []
            for item_id in click_history:
                item_datum = self.meta_data[item_id]
                item_title = 'unknown title'
                if 'app_name' in item_datum:
                    item_title = item_datum['app_name']
                history_titles.append(item_title)
            
            target_item_datum = self.meta_data[target_item]
            target_item_title = 'unknown title'
            if 'app_name' in target_item_datum:
                target_item_title = target_item_datum['app_name']

            task_template = self.all_task_templates['explanation']
            source_text = task_template['source'].format(', '.join(history_titles), target_item_title)
            target_text = task_template['target'].format("No ground truth.")
            source_text = re.sub('[^A-Za-z0-9_.,!?;:\n ]', '', source_text)
            target_text = re.sub('[^A-Za-z0-9_.,!?;:\n ]', '', target_text)
            history_titles = re.sub('[^A-Za-z0-9_.,!?;:\n ]', '', ', '.join(history_titles))
            target_item_title = re.sub('[^A-Za-z0-9_.,!?;:\n ]', '', target_item_title)
            datasets.append({
                "source": source_text,
                "history": history_titles,
                "target": target_item_title,
                "task": "explanation"
            })
        return datasets
    def gen_conversation_data(self, sample_num):
        datasets = []
        sample_num = min(sample_num, len(self.sequential_data))
        for idx in range(sample_num):
            exp_datum = self.sequential_data[idx]
            sequence = [int(x) for x in exp_datum.split()]
            click_history = sequence[1:-1]
            target_item = sequence[-1]
            history_titles = []
            for item_id in click_history:
                item_datum = self.meta_data[item_id]
                item_title = 'unknown title'
                if 'app_name' in item_datum:
                    item_title = item_datum['app_name']
                history_titles.append(item_title)
            
            target_item_datum = self.meta_data[target_item]
            target_item_title = 'unknown title'
            if 'app_name' in target_item_datum:
                target_item_title = target_item_datum['app_name']

            history_titles = re.sub('[^A-Za-z0-9_.,!?;:\n ]', '', ', '.join(history_titles))
            target_item_title = re.sub('[^A-Za-z0-9_.,!?;:\n ]', '', target_item_title)
            datasets.append({
                "history": history_titles,
                "target": target_item_title,
                "task": "conversation"
            })
        return datasets

    def gen_search_data(self, sample_num):
        datasets = []
        if self.search_data is None:
            return datasets

        sample_num = min(sample_num, len(self.search_data))
        for idx in range(sample_num):
            target_item_title = self.search_data['target'][idx]
            response = self.search_data['response'][idx]
            queries = response.strip().split(",")
            test_query = random.sample(queries, 1)[0]

            task_template = self.all_task_templates['search']
            source_text = task_template['source'].format(test_query)
            target_text = task_template['target'].format(target_item_title)
            source_text = re.sub('[^A-Za-z0-9_.,!?;:\n ]', '', source_text)
            target_text = re.sub('[^A-Za-z0-9_.,!?;:\n ]', '', target_text)
            datasets.append({
                "source": source_text,
                "target": target_text,
                "task": "searching"
            })
        return datasets    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=str, default='ranking,retrieval,explanation,conversation', help='tasks for data generation.')
    parser.add_argument('--prompt_type', type=str, default='neutral')
    parser.add_argument('--sample_num', type=int, default=1000, help='sample number for each task.')  
    parser.add_argument('--dataset', type=str, default='steam', help='the dataset to be evaluated, steam/beauty/sports')
    parser.add_argument('--version_num', type=str)
    # parser.add_argument("--split", type=str, default="train", help="Dataset split (train/val/test)") 

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if args.dataset == 'steam':    
        from all_steam_templates import all_tasks as all_task_templates

    elif args.dataset == "ml-1m":
        if args.prompt_type == "neutral":
            from nonsensitive_ml1m_templates import all_tasks as all_task_templates
        elif args.prompt_type == "sensitive":
            from sensitive_ml1m_templates import all_tasks as all_task_templates

    elif args.dataset == "lfm-1b":
        if args.prompt_type == "neutral":
            from nonsensitive_lfm1b_templates import all_tasks as all_task_templates
        elif args.prompt_type == "sensitive":
            from sensitive_lfm1b_templates import all_tasks as all_task_templates

    elif args.dataset == "jobrec":
        if args.prompt_type == "neutral":
            from nonsensitive_jobrec_templates import all_tasks as all_task_templates
        elif args.prompt_type == "sensitive":
            from sensitive_jobrec_templates import all_tasks as all_task_templates

    
    dataset = Test_Dataset(all_task_templates, dataset=args.dataset, prompt_type=args.prompt_type)

    if "retrieval" in args.tasks:
        print(f'generating retrieval data {args.dataset}, sample number: {args.sample_num}, prompt type: {args.prompt_type}, version: {args.version_num} ...')
        data = dataset.gen_retrieval_data(args.sample_num)
        fd = open(f"data/{args.dataset}/retrieval_{args.prompt_type}_{args.version_num}.jsonl", "w")

        if args.dataset == "steam":
            for line in data:
                line = {
                    "id": line["id"],
                    "prompt": line["source"],
                    "target": line["target"],
                    "history": line["history"],
                    "task": line["task"],
                }
                fd.write(json.dumps(line)+'\n')
        
        else:
            for line in data:
                line = {
                    "id": line["id"],
                    "prompt": line["source"],
                    "target": line["target"],
                    "history": line["history"],
                    "val": line["val"],
                    "task": line["task"],
                }
                fd.write(json.dumps(line)+'\n')

    if "ranking" in args.tasks:
        print(f'generating ranking data, sample number: {args.sample_num} ...')
        data = dataset.gen_ranking_data(args.sample_num)
        fd = open(f"data/{args.dataset}/ranking.jsonl", "w")
        for line in data:
            line = {
                "prompt": line["source"],
                "target": line["target"],
                "history": line["history"],
                "candidate": line["candidate"],
                "task": line["task"],
            }
            fd.write(json.dumps(line)+'\n')
    if "explanation" in args.tasks:
        print(f'generating explanation data, sample number: {args.sample_num} ...')
        data = dataset.gen_explanation_data(args.sample_num)
        fd = open(f"data/{args.dataset}/explanation.jsonl", "w")
        for line in data:
            line = {
                "prompt": line["source"],
                "history": line["history"],
                "target": line["target"],
                "task": line["task"],
            }
            fd.write(json.dumps(line)+'\n')
    if "conversation" in args.tasks:
        print(f'generating conversation data, sample number: {args.sample_num} ...')
        data = dataset.gen_conversation_data(args.sample_num)
        fd = open(f"data/{args.dataset}/conversation.jsonl", "w")
        for line in data:
            line = {
                "history": line["history"],
                "target": line["target"],
                "task": line["task"],
                "user_simulator_system_prompt": "You are a user chatting with a recommender for recommendation in turn. Your history is {history}. Your target items: {target}.\nYou must follow the instructions below during chat.\nIf the recommender recommends {target}, you should accept.\nIf the recommender recommends other items, you should refuse them and provide the information about {target}. You should never directly tell the target item title.\nIf the recommender asks for your preference, you should provide the information about {target}. You should never directly tell the target item title.\nNow lets start, you first, act as a user.\n Your output is only allowed to be the words from the user you act.".format(
                    history=line["history"],
                    target=line["target"],
                )
            }
            fd.write(json.dumps(line)+'\n')
    if "chatbot" in args.tasks:  # use previous 3 tasks' data
        print(f'generating chatbot data, sample number: {args.sample_num} ...')

        data1 = dataset.gen_explanation_data(args.sample_num // 3)
        data2 = dataset.gen_ranking_data(args.sample_num // 3)
        data3 = dataset.gen_retrieval_data(args.sample_num // 3)

        combined_data = data1 + data2 + data3
        if len(combined_data) > args.sample_num:
            combined_data = combined_data[:args.sample_num]  
        elif len(combined_data) < args.sample_num:
            additional_data = dataset.gen_explanation_data(args.sample_num - len(combined_data))
            combined_data += additional_data

        fd = open(f"data/{args.dataset}/chatbot.jsonl", "w")
        for line in combined_data:
            line_to_write = {
                "prompt": line["source"],
                "task": "chatbot",
            }
            fd.write(json.dumps(line_to_write) + '\n')
        fd.close()

        print(f"Successfully generated {len(combined_data)} samples for chatbot.")