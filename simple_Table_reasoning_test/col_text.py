import time
import json
import argparse
import copy
import os

from typing import List
import platform
import multiprocessing

from generation.generator_gpt import Generator
from utils.utils import load_data_split
from nsql.database import NeuralDB

from transformers import AutoTokenizer
import re

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, default="results/model_gpt")
parser.add_argument(
    "--input_program_file", type=str, default="wikitq_test_test_result.json"
)
args = parser.parse_args()

with open(os.path.join(args.save_dir, args.input_program_file), "r") as f:
    data = json.load(f)  # 上一步col_sql生成的result文件，包亏了原始数据和generation信息
    col_dict = dict()
    for eid, _ in data.items():
        data_item = data[eid]["ori_data_item"]
        if data[eid]["generations"]:
            col_gen = data[eid]["generations"]
        else:
            col_gen = []
        col_dict[eid] = {"output": col_gen, "data_item": data_item}
        #dayin
        # print(col_dict[eid]["output"])
        print(col_dict[eid]["output"])
        # 上一层的信息(generations和ori_data_item)转换为每个eid对应的output和data_item
        pattern_col = "(f_col\(\[(.*?)\]\))"
        pattern_col = re.compile(pattern_col, re.S)

        g_eids=[0]

        for g_eid in g_eids:
        try:
            # Extract Columns
            preds=[]
            for n in range(args.sampling_n):
                try:
                    pred = re.findall(pattern_col,col_dict[str(g_eid)]['cols'][n])[0][1]
                    pred = pred.split(', ')
                    new_pred = []
                    for i in pred:
                        if i.startswith("'"):
                            i = i[1:]

                        if i.endswith("'"):
                            i = i[:-1]
                        new_pred.append(i)
                    print(new_pred)
                    preds.append(new_pred)
                except:
                    pass
            pred = list(set().union(*preds,col_dict[str(g_eid)]['output']))
            g_data_item = col_dict[str(g_eid)]['data_item']
            g_dict[g_eid] = {
                'generations': [],
                'cols' : [],
                'ori_data_item': copy.deepcopy(g_data_item)
            }
