#===============--generate_type是'col'而不是answer 因为这一层想要的答案是最终SQL语句

import time
import json
import argparse
import copy
import os
import regex as re
from nsql.sql_exec import Executor, extract_answers
from typing import List
import platform
import multiprocessing
import tiktoken
import pandas as pd
from generation.generator_gpt_2 import Generator
from simple_Table_reasoning_test.sample import extract
from utils.utils import load_data_split
from nsql.database import NeuralDB
from nsql.parser import extract_rows
from utils.normalizer import post_process_sql
from nsql.sql_exec import Executor, extract_rows
import numpy as np

def worker_annotate(
        pid: int,
        args,
        keys: list,
        g_eids: List,
        data,
        tokenizer
):
    generator = Generator(args, keys=keys)
    """
    A worker process for annotating.
    """
    g_dict = dict()
    built_few_shot_prompts = []

    pattern_row = '(f_row\(\[(.*?)\]\))'
    pattern_row_num = '\d+'
    pattern_row = re.compile(pattern_row, re.S)
    pattern_row_num = re.compile(pattern_row_num, re.S)

    # Extract Rows
    for g_eid in g_eids:
        try:
            preds=[]
            for n in range(1):
                try:
                    pred = re.findall(pattern_row,data[str(g_eid)]['output'][n])[0][1]
                    if pred == '*':
                        pred = ''
                        for i in range(len(data[str(g_eid)]['data_item']['table']['rows'])):
                            pred += f'row {i}'
                            if i != len(data[str(g_eid)]['data_item']['table']['rows'])-1:
                                pred += ', '

                        pred = pred.split(', ')
                        preds.append(pred)
                    else:
                        pred = pred.split(', ')
                        preds.append(pred)
                except:
                    pred = data[str(g_eid)]['rows'] 
                    if pred == [] or '':
                        pred = ''
                        for i in range(len(data[str(g_eid)]['data_item']['table']['rows'])):
                            pred += f'row {i}'
                            if i != len(data[str(g_eid)]['data_item']['table']['rows'])-1:
                                pred += ', '
                        pred = pred.split(', ')
                    preds.append(pred)

            # pred = list(set().union(*preds, data[str(g_eid)]['rows']))
            all_preds = preds + [data[str(g_eid)]['rows']]
            pred = set().union(*all_preds)
            t=[data[str(g_eid)]['rows']]
            print(f'解析出来的preds是{preds} 之前的解析好的是{t}')

            # print(f' 解析的preds是{preds} \n ')
            # print(f'最后过滤的所有row是{preds}')
            
            g_data_item = data[str(g_eid)]['data_item']
            g_dict[g_eid] = {
                'generations': [],
                'cols' : [],
                'rows' : [],
                'ori_data_item': copy.deepcopy(g_data_item)
            }
            db = NeuralDB(
                tables=[{'title': g_data_item['table']['page_title'], 'table': g_data_item['table']}]
            )
            g_data_item['table'] = db.get_table_df()
            g_data_item['title'] = db.get_table_title()

            df = g_data_item['table']

            # Filter Columns
            filtered_cols = [value for value in g_data_item['table'].columns if value in data[str(g_eid)]['cols']]
            if filtered_cols == []:
                filtered_cols = [value for value in g_data_item['table'].columns]
            g_dict[g_eid]['cols'] = filtered_cols

            if [data[str(g_eid)]['cols']] != []:
                if 'row_id' not in filtered_cols:
                    filtered_cols.insert(0, 'row_id')
                df = g_data_item['table'][filtered_cols]
            
            if pred != [""]:
                row_list = [str(pattern_row_num.search(x).group()) for x in pred if pattern_row_num.search(x)]
                if row_list == []:
                    row_list = [str(j) for j in range(len(g_data_item['table']['rows']))]
                    row_unique = set(row_list)
                    row_list = list(row_unique)
                else:
                    row_unique = set(row_list)
                    row_list = list(row_unique)
                    try:
                        indices = [eval(i) for i in row_list]
                        if any(index >= len(df) for index in indices):
                            raise IndexError("Index out of bounds")
                        df = df[df.index.isin(indices)]
                    except IndexError:
                        df = df

            if pred == [] or pred == [""]:
                row_list = [str(j) for j in range(len(g_data_item['table']['rows']))]
                row_unique = set(row_list)
                row_list = list(row_unique)
                
                df = df[df.index.isin([eval(i) for i in row_list])]
            
            g_dict[g_eid]['rows'] = row_list
            g_data_item['table'] = df

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
            
            print(generate_prompt)
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
            print(f"Process#{pid}: Building prompt for eid#{g_eid}, original_id#{g_data_item['id']}")
            built_few_shot_prompts.append((g_eid, prompt))
            if len(built_few_shot_prompts) < args.n_parallel_prompts:
                continue

            print(f"Process#{pid}: Prompts ready with {len(built_few_shot_prompts)} parallels. Run openai API.")
            response_dict = generator.generate_one_pass(
                prompts=built_few_shot_prompts,
                verbose=args.verbose
            )
            for eid, g_pairs in response_dict.items():
                
                g_dict[eid]['generations'] = g_pairs
            
            built_few_shot_prompts = []
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Process#{pid}: eid#{g_eid}, wtqid#{g_data_item['id']} generation error: {e}")

    # # Final generation inference
    # if len(built_few_shot_prompts) > 0:
    #     response_dict = generator.generate_one_pass(
    #         prompts=built_few_shot_prompts,
    #         verbose=args.verbose
    #     )
    #     for eid, g_pairs in response_dict.items():
            
    #         g_dict[eid]['generations'] = g_pairs
    
    return g_dict


def main():
    start_time = time.time()
    with open(args.api_keys_file, "r") as f:
        keys = [line.strip() for line in f if line.strip()]

    generator = Generator(args, keys=keys)

#读取原始数据===================================
    dataset=extract()#采样15个数据
    with open(os.path.join(args.save_dir, args.input_program_file), 'r') as f:
        data = json.load(f)
    data_dict = dict()
    for eid, _ in data.items():
        data_item = data[eid]['ori_data_item']
        if data[eid]['generations'] or data[eid]['cols']:
            output = data[eid]['generations']
            rows = data[eid]['rows']
            cols = data[eid]['cols']
        else:
            rows = []
            cols = []
        data_dict[eid] = {'rows': rows, 'cols': cols, 'output' : output, 'data_item' : data_item}
    
    # Split by processes
    data_dict_group = [dict() for _ in range(args.n_processes)]
    # for idx, eid in enumerate(data_dict.keys()):
    #     data_dict_group[idx % args.n_processes][eid] = data_dict[eid]
    for idx, eid in enumerate(sorted(data_dict.keys(), key=int)):  # 按整数排序
        data_dict_group[idx % args.n_processes][eid] = data_dict[eid]
    
    generate_eids = list(range(0,15))
    
    generate_eids_group = [[] for _ in range(args.n_processes)]
    for g_eid in generate_eids:
        generate_eids_group[int(g_eid) % args.n_processes].append(g_eid)
    
    print('\n******* Annotating *******')
    g_dict = dict()
    worker_results = []
    pool = multiprocessing.Pool(processes=args.n_processes)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    for pid in range(args.n_processes):
        
        
        worker_results.append(pool.apply_async(worker_annotate, args=(
            pid,
            args,
            keys,
            generate_eids_group[pid],
            data_dict_group[pid],
            tokenizer
        )))

    # Merge annotation results
    for r in worker_results:
        worker_g_dict = r.get()
        g_dict.update(worker_g_dict)
    pool.close()
    pool.join()
    save_file_name='reason_sql111.json'
    with open(os.path.join(args.save_dir, save_file_name), 'w') as f:
        json.dump(g_dict, f, indent=4)

    print(f"Elapsed time: {time.time() - start_time}")
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_keys_file", type=str, default="key.txt")
    parser.add_argument("--prompt_file", type=str, default="./prompts/sql_reason_wtq.txt")

    parser.add_argument(
        "--generate_type",
        type=str,
        default="col",
        choices=["col", "answer", "row", "verification"],
    )
    parser.add_argument("--n_shots", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    # LLM options
    parser.add_argument(
        "--engine", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct"
    )  # Qwen/Qwen2.5-7B-Instruct#Qwen/Qwen2.5-VL-32B-Instruct
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
    parser.add_argument(
        "--dataset", type=str, default="fetaqa", choices=["wikitq", "tab_fact", "fetaqa"]
    )
    # 最关键的3个路径，promptfile和save_dir，input_program_file
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="create_table_select_full_table",
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
    parser.add_argument("--save_dir", type=str, default="results/multi_process_model_gpt")
    parser.add_argument("--n_processes", type=int, default=5)
    parser.add_argument("--input_program_file", type=str, default="row_text.json")

    # 先和上一层一样 先加载原始的表格数据和API KEYS=========================================================
    args = parser.parse_args()
    args.dataset_path = f"./datasets/data/fetaQA-v1_test.json"

    main()
