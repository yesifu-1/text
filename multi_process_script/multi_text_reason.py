#===============--generate_type是'col'而不是answer

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
from utils.utils import load_data_split
from nsql.database import NeuralDB
from nsql.parser import extract_rows
from utils.normalizer import post_process_sql
from nsql.sql_exec import Executor, extract_rows
import numpy as np
from simple_Table_reasoning_test.sample import extract

def worker_annotate(
        pid: int,
        args,
        keys: list,
        g_eids: List,
        row_col,
        tokenizer
):
    """
    A worker process for annotating.
    """
    generator = Generator(args, keys=keys)
    g_dict = dict()
    built_few_shot_prompts = []

    pattern_row = '(f_row\(\[(.*?)\]\))'
    pattern_row_num = '\d+'
    pattern_row = re.compile(pattern_row, re.S)
    pattern_row_num = re.compile(pattern_row_num, re.S)
    
    for g_eid in g_eids:
        try:
            executor = Executor(args, "")
            g_data_item = row_col[str(g_eid)]['data_item']
            g_dict[g_eid] = {
                'generations': [],
                'ori_data_item': copy.deepcopy(g_data_item)
            }

            db = NeuralDB(
                tables=[{'title': g_data_item['table']['page_title'], 'table': g_data_item['table']}]
            )
            tab = db.get_table_df()
            g_data_item['title'] = db.get_table_title()
            df = tab

            ### Filter columns ###
            filtered_cols = [value for value in tab.columns if value in row_col[str(g_eid)]['cols']]

            if filtered_cols == []:
                filtered_cols = [value for value in tab.columns]
            if 'row_id' in filtered_cols:
                filtered_cols.pop(0)
            g_dict[g_eid]['cols'] = filtered_cols

            if [row_col[str(g_eid)]['cols']] != []:
                df = tab[filtered_cols]

            ### Filter rows ###
            if row_col[str(g_eid)]['rows'] == []:
                row_list = [str(j) for j in range(len(g_data_item['table']['rows']))]
                row_unique = set(row_list)
                row_list = list(row_unique)
                df = df[df.index.isin([eval(i) for i in row_list])]

            else:
                row_list = [str(pattern_row_num.search(x).group()) for x in row_col[str(g_eid)]['rows'] if pattern_row_num.search(x)]
                row_unique = set(row_list)
                row_list = list(row_unique)
                try:
                    indices = [eval(i) for i in row_list]
                    if any(index >= len(df) for index in indices):
                        raise IndexError("Index out of bounds")
                    df = df[df.index.isin(indices)]
                except IndexError:
                    df = df

            db = NeuralDB(
                tables=[{'title': g_data_item['title'], 'table': g_data_item['table']}]
            )
            g_data_item['table'] = df

            # Additional Evidence
            string = ''
            try:
                for n in range(1):
                    try:
                        nsql = row_col[str(g_eid)]['output'][n]
                        sql = nsql.split('SQL: ')[1]
                        norm_sql = post_process_sql(
                        sql_str=sql,
                        df=g_data_item['table'],
                        process_program_with_fuzzy_match_on_db=True,
                        table_title= g_data_item['title']
                        )
                        exec_answer = executor.sql_exec(norm_sql, db, verbose=False)
                        exec_rows = extract_answers(exec_answer)
                        # Convert dictionary to DataFrame
                        new_df = pd.DataFrame(exec_answer["rows"], columns=exec_answer["header"])
                        if 'row_id' in new_df.columns.tolist():
                            new_df.drop(columns = 'row_id', inplace = True)
                        if 'index' in new_df.columns.tolist():
                            new_df.drop(columns = 'index', inplace = True)
                        if np.array_equal(new_df.values,df.values):
                            continue

                        string = 'Here is an additional evidence to help the answering process.\nAdditional Evidence:\n/*\n'
                        string += 'col : ' + ' | '.join(new_df.columns) + '\n'
                        for row_id, row in new_df.iloc[:len(new_df)].iterrows():
                            string += f'row {row_id} : '
                            for column_id, header in enumerate(new_df.columns):
                                string += str(row[header])
                                if column_id != len(new_df.columns) - 1:
                                    string += ' | '
                            string += '\n'
                        string += '*/\n'
                    
                    except:
                        pass
            except:
                pass

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
            if string:
                prompt += '\n' + string
                print(generate_prompt, string)
            else:
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
                print('damn!!!!!!!!!!!!!!!!!')
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
    #         g_pairs = sorted(g_pairs, key=lambda x: x[-1], reverse=True)
    #         g_dict[eid]['generations'] = g_pairs

    return g_dict


def main():
    start_time = time.time()
    with open(args.api_keys_file, "r") as f:
        keys = [line.strip() for line in f if line.strip()]

    generator = Generator(args, keys=keys)
    dataset=extract()#采样15个数据
#读取input数据===================================
    with open(os.path.join(args.save_dir, args.input_program_file), 'r') as f:
        data = json.load(f)
    row_col_dict = dict()
    for eid, _ in data.items():
        data_item = data[eid]['ori_data_item']
        if data[eid]['generations'] or data[eid]['cols'] or data[eid]['rows']:
            output = data[eid]['generations']
            rows = data[eid]['rows']
            cols = data[eid]['cols']
        else:
            rows = []
            cols = []
        row_col_dict[eid] = {'rows': rows, 'cols': cols, 'output' : output, 'data_item' : data_item}

    # Split by processes
    row_col_dict_group = [dict() for _ in range(args.n_processes)]
    # for idx, eid in enumerate(row_col_dict.keys()):
    #     row_col_dict_group[idx % args.n_processes][eid] = row_col_dict[eid]
    for idx, eid in enumerate(sorted(row_col_dict.keys(), key=int)):  # 按整数排序
        row_col_dict_group[idx % args.n_processes][eid] = row_col_dict[eid]


    # Annotate
    generator = Generator(args, keys=keys)
    # Enter dataset size for inference: range(0, len(dataset))
    generate_eids = list(range(0,len(dataset)))

    generate_eids_group = [[] for _ in range(args.n_processes)]
    for g_eid in generate_eids:
        generate_eids_group[int(g_eid) % args.n_processes].append(g_eid)
    print('\n******* Annotating *******')
    # if 'gpt' or 'bison' in args.engine:
    #     tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(ROOT_DIR, "utils", "gpt2"))
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.engine)
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
            row_col_dict_group[pid],
            tokenizer
        )))

    # Merge annotation results
    for r in worker_results:
        worker_g_dict = r.get()
        g_dict.update(worker_g_dict)
    pool.close()
    pool.join()
    model_name=args.engine.split('/')[1]
    
#==================================保存结果=========\
    save_file_name = f"{args.dataset}_{model_name}_reason_text.json"
    os.makedirs(args.save_dir, exist_ok=True)

    with open(os.path.join(args.save_dir, save_file_name),"w") as f:
        json.dump(g_dict, f, indent=2)


    print(">> 测试完成")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_keys_file", type=str, default="key.txt")
    parser.add_argument("--prompt_file", type=str, default="./prompts/fetaqa.txt")

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
    parser.add_argument("--input_program_file", type=str, default="reason_sql.json")
    parser.add_argument("--n_processes", type=int, default=5)

    # 先和上一层一样 先加载原始的表格数据和API KEYS=========================================================
    args = parser.parse_args()
    args.dataset_path = f"./datasets/data/fetaQA-v1_test.json"
    main()