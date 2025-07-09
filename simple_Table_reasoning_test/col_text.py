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
import tiktoken

from transformers import AutoTokenizer
import re
import pandas as pd

from generation.generator_gpt_2 import Generator
parser = argparse.ArgumentParser()

parser.add_argument("--api_keys_file", type=str, default="key.txt")
parser.add_argument("--prompt_file", type=str, default="./prompts/col_select_text.txt")

parser.add_argument(
    "--generate_type",
    type=str,
    default="col",
    choices=["col", "answer", "row", "verification"],
)
parser.add_argument("--n_shots", type=int, default=8)
parser.add_argument("--seed", type=int, default=42)

# LLM options
parser.add_argument("--engine", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct")#Qwen/Qwen2.5-7B-Instruct#Qwen/Qwen2.5-VL-32B-Instruct
parser.add_argument("--n_parallel_prompts", type=int, default=1)
parser.add_argument("--max_generation_tokens", type=int, default=2000)
parser.add_argument("--max_api_total_tokens", type=int, default=180000)
parser.add_argument("--temperature", type=float, default=0.4)
parser.add_argument("--sampling_n", type=int, default=1)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument(
    "--stop_tokens", type=str, default="\n\n", help="Split stop tokens by ||"
)
# debug options
parser.add_argument("-v", "--verbose", action="store_false")
parser.add_argument('--dataset', type=str, default='fetaqa',
                        choices=['wikitq', 'tab_fact', 'fetaqa'])
#最关键的3个路径，promptfile和save_dir，input_program_file
parser.add_argument(
    "--prompt_style",
    type=str,
    default="transpose",
    choices=[
        "create_table_select_3_full_table",
        "create_table_select_full_table",
        "create_table_select_3",
        "create_table",
        "text_full_table",
        "transpose",
        "no_table",
    ],
)
parser.add_argument("--save_dir", type=str, default="results/model_gpt")
parser.add_argument(
    "--input_program_file", type=str, default="fetaqa_col_sql.json"
)
#先和上一层一样 先加载原始的表格数据和API KEYS=========================================================
args = parser.parse_args()
args.dataset_path = f"./datasets/data/fetaQA-v1_test.json"

with open(args.dataset_path, "r") as f:
    dataset = json.load(f)

with open(args.api_keys_file, "r") as f:
    keys = [line.strip() for line in f if line.strip()]

generator = Generator(args, keys=keys)

tokenizer = tiktoken.get_encoding("cl100k_base")
if generator and tokenizer:
    print(f"✓ 成功初始化生成器和分词器")
else:
    print("❌ 生成器或分词器初始化失败，请检查配置")

#这里接收了上一步生成的col_sql的json文件=========================================

with open(os.path.join(args.save_dir, args.input_program_file), "r") as f:
    data = json.load(f)  # 上一步col_sql生成的result文件，包亏了原始数据ori_data_item和生成的generation信息 ，这里用data_item和output分别接收。
col_dict = dict()#col_dic是部分要送进LLM的数据(还需要后续处理)
for eid, _ in data.items():
    data_item = data[eid]["ori_data_item"]
    if data[eid]["generations"]:
        col_gen = data[eid]["generations"]
    else:
        col_gen = []
    col_dict[eid] = {"output": col_gen, "data_item": data_item}
    
    
    # 上一层的信息(generations和ori_data_item)转换为每个eid对应的output和data_item


    #进入处理逻辑，g_data_item 是要输入给LLM的数据，g_dict是最后用于接收结果的字典(要传入下一层的内容)==========================================================
    g_dict = dict()
    g_eids=[0]
    built_few_shot_prompts = []
    pattern_col = "(f_col\(\[(.*?)\]\))"
    pattern_col = re.compile(pattern_col, re.S)

    

    g_eid=0#test
    preds=[]
    pred_col = ""

    pred_cols=[]
    new_df = {}
    # Extract Columns
    for n in range(2):
        try:
            pred_col = re.findall(pattern_col,col_dict[str(g_eid)]['output'][n])[0][1]#col_dict是json文件 所以要进行str(g_eid)
            
            pred_col = pred_col.replace("'","")
            pred_col = pred_col.split(', ')
            pred_cols.append(pred_col)
            print(f'第一步 col_sql得到的与问题相关的列是{pred_cols}，准备开始第二部col_test,转置表格来看哪些列对于问题重要===========================')
            
        except:
            pass
    pred_col = list(set().union(*pred_cols))
    g_data_item = col_dict[str(g_eid)]['data_item']
   
    g_dict[g_eid] = {
        'generations': [],
        'cols' : [],
        'ori_data_item': copy.deepcopy(g_data_item)
    }
   
    #随后LLM处理完的结果放到g_dict的cols和generation键中

    rows = g_data_item['table']['rows']#获取原始表格数据的行
    header = g_data_item['table']['header']#获取原始表格数据的列

    df = pd.DataFrame(rows, columns=header)#把列表形式表示的表格数据转换为 DataFrame
    g_data_item['title'] = g_data_item['table']['page_title']
    g_data_item['table'] = df#覆盖掉了
    
    
    #原始表格数据的保存 
    g_dict[g_eid]['ori_data_item']['table']['header']= df.columns.tolist()
    g_dict[g_eid]['ori_data_item']['table']['rows']= df.values.tolist()
    
    if pred_col != []:
                filtered_pred = [value for value in pred_col if value in header]
                g_dict[g_eid]['cols'] = filtered_pred

    #pred_col中是第一部col_sql预测的相关列名


    #使用提示将表格转置========================================================

    n_shots = args.n_shots
    few_shot_prompt = generator.build_few_shot_prompt_from_file(
        file_path=args.prompt_file,
        n_shots=n_shots
    )
    
    generate_prompt = generator.build_generate_prompt(
                data_item=g_data_item,
                generate_type=(args.generate_type,)
            )
    
    prompt = few_shot_prompt + "\n\n" + generate_prompt
    # Ensure the input length fit max input tokens by shrinking the number of rows
    max_prompt_tokens = args.max_api_total_tokens - args.max_generation_tokens
    
    num_rows = (g_data_item['table'].shape[0])
    while len(tokenizer.encode(prompt)) >= max_prompt_tokens:
        num_rows = 5
        generate_prompt = generator.build_generate_prompt(
        data_item=g_data_item,
        generate_type=(args.generate_type,),
        num_rows = num_rows
        )

        prompt = few_shot_prompt + "\n\n" + generate_prompt
    print(f"Process#某个线程: Building prompt for eid#{g_eid}, original_id#{g_data_item['id']}")
    built_few_shot_prompts.append((g_eid, prompt))#对单个表格无意义

#=================调用模型把表转置==================
    start_time = time.time()
    response_dict = generator.generate_one_pass(
    prompts=[(eid, prompt),(str(int(eid)+1),prompt)], verbose=args.verbose)
    print(f">> 耗时：{time.time() - start_time:.2f} 秒")
    
    response=[]
    for (i,res) in response_dict.items():
        response.extend(res)
   #json中读出的数据eid--->for eid, _ in data.items():   eid在json字符串中被转换成了str，手动转换回数字
    g_dict[int(eid)]['generations'] = response
    
    for i, text in enumerate(response):
        print(f"\n[回答 {i + 1}] : {text}\n")
        
#====================存储结果=====================================
    os.makedirs(args.save_dir, exist_ok=True)
    with open(
        os.path.join(
            args.save_dir, f"{args.dataset}_col_text.json"
        ),
        "w",
    ) as f:
        json.dump(g_dict, f, indent=2)

    print(">> 测试完成")
