import os
import time
import json
import pandas as pd
from transformers import AutoTokenizer
import tiktoken
import argparse
from generation.generator_gpt_2 import Generator  # 核心生成器类
from simple_Table_reasoning_test.sample import extract
import copy
import multiprocessing
from nsql.database import NeuralDB
import re

def worker_annotate(
        pid: int,
        args,
        keys: list,
        g_eids: list,
        col_dict,
        tokenizer
):

    generator = Generator(args, keys=keys)
    g_dict = dict()
    built_few_shot_prompts = []
    pattern_col = '(f_col\(\[(.*?)\]\))'
    pattern_col = re.compile(pattern_col, re.S)

    for g_eid in g_eids:
        pred_col = ""
        try:
            pred_cols=[]
            new_df = {}
            # Extract Columns
            for n in range(2):
                try:
                    pred_col = re.findall(pattern_col,col_dict[str(g_eid)]['output'][n])[0][1]#col_dict是json文件 所以要进行str(g_eid)
                    pred_col = pred_col.replace("'","")
                    pred_col = pred_col.split(', ')
                    pred_cols.append(pred_col)
                    print(pred_cols)

                except:
                    pass
            pred_col = list(set().union(*pred_cols))
            g_data_item = col_dict[str(g_eid)]['data_item']
            g_dict[g_eid] = {
                'generations': [],
                'cols' : [],
                'ori_data_item': copy.deepcopy(g_data_item)
            }
            df = pd.DataFrame(data=g_data_item['table']['rows'], columns=g_data_item['table']['header'])
            if new_df is not None and len(new_df) == len(df):
                df = df.join(new_df)

            g_dict[g_eid]['ori_data_item']['table']['header']= df.columns.tolist()
            g_dict[g_eid]['ori_data_item']['table']['rows']= df.values.tolist()

            db = NeuralDB(
                tables=[{'title': g_data_item['table']['page_title'], 'table': g_dict[g_eid]['ori_data_item']['table']}]
            )
            
            g_data_item['table'] = db.get_table_df()
            g_data_item['title'] = db.get_table_title()
            if pred_col != []:
                filtered_pred = [value for value in pred_col if value in g_data_item['table'].columns]#疑似错误 
                g_dict[g_eid]['cols'] = filtered_pred


            n_shots = args.n_shots
            few_shot_prompt = generator.build_few_shot_prompt_from_file(
                file_path=args.prompt_file,
                n_shots=n_shots
            )
            # Take table transpose
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
                g_pairs = sorted(g_pairs, key=lambda x: x[-1], reverse=True)
                g_dict[eid]['generations'] = g_pairs
            
            built_few_shot_prompts = []
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Process#{pid}: eid#{g_eid}, wtqid#{g_data_item['id']} generation error: {e}")

    # Final generation inference
    if len(built_few_shot_prompts) > 0:
        response_dict = generator.generate_one_pass(
            prompts=built_few_shot_prompts,
            verbose=args.verbose
        )
        for eid, g_pairs in response_dict.items():
            g_pairs = sorted(g_pairs, key=lambda x: x[-1], reverse=True)
            g_dict[eid]['generations'] = g_pairs
    
    return g_dict


def main():
    # ==== 加载表格数据（JSON） ====
    
    start_time = time.time()
    dataset=extract()
    
        
        

    # ==== 加载 API keys ====
    with open(args.api_keys_file, "r") as f:
        keys = [line.strip() for line in f if line.strip()]

  #================================================
#很重要 这里接收了上一步生成的col_sql的json文件
    with open(os.path.join(args.save_dir, args.input_program_file), 'r') as f:
        data = json.load(f)
    col_dict = dict()
    for eid, _ in data.items():
        data_item = data[eid]['ori_data_item']
        if data[eid]['generations']:
            col_gen = data[eid]['generations']
        else:
            col_gen = []
        col_dict[eid] = {'output': col_gen, 'data_item': data_item}
#================================================
    # Split by processes
    col_dict_group = [dict() for _ in range(args.n_processes)]
    # for idx, eid in enumerate(col_dict.keys()):
    #     col_dict_group[idx % args.n_processes][eid] = col_dict[eid]
    for idx, eid in enumerate(sorted(col_dict.keys(), key=int)):  # 按整数排序
        col_dict_group[idx % args.n_processes][eid] = col_dict[eid]

    
    # Annotate
    print(len(col_dict))
    generator = Generator(args, keys=keys)
    # Enter dataset size for inference: range(0, len(dataset))
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
            col_dict_group[pid],
            tokenizer
        )))

    # Merge annotation results
    for r in worker_results:
        worker_g_dict = r.get()
        g_dict.update(worker_g_dict)
    pool.close()
    pool.join()

    save_file_name = 'col_text.json'
    with open(os.path.join(args.save_dir, save_file_name), 'w') as f:
        json.dump(g_dict, f, indent=4)

    print(f"Elapsed time: {time.time() - start_time}")




if __name__=='__main__':
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
    parser.add_argument("--engine", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct") #Qwen/Qwen2.5-7B-Instruct #Qwen/Qwen2.5-VL-32B-Instruct #没有shen
    parser.add_argument("--n_parallel_prompts", type=int, default=2)
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
    parser.add_argument("--save_dir", type=str, default="results/multi_process_model_gpt")
    parser.add_argument(
        "--input_program_file", type=str, default="col_sql.json"
    )
    #先和上一层一样 先加载原始的表格数据和API KEYS=========================================================
    args = parser.parse_args()
    args.n_processes=5
    args.dataset_path = f"./datasets/data/fetaQA-v1_test.json"
    
    main()