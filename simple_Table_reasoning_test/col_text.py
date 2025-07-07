import re
import argparse
import os
import json

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

        pred_col = ""
