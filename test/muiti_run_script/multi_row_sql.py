"""
SQL based Row Extraction
注意:最后合并筛选的行的sql条件时，用的逻辑居然是or而不是and
"""

import time
import json
import argparse
import copy
import os
import regex as re

from typing import List
import platform
import multiprocessing
import tiktoken
from generation.generator_gpt_2 import Generator
from simple_Table_reasoning_test.sample import extract
from utils.utils import load_data_split
from nsql.database import NeuralDB
from nsql.parser import extract_rows
from utils.normalizer import post_process_sql
from nsql.sql_exec import Executor, extract_rows
def worker_annotate(
        pid: int,
        args,
        keys: list,
        g_eids: List,
        col_dict,
        tokenizer
):
    """
    A worker process for annotating.
    """
    generator = Generator(args, keys=keys)
    g_dict = dict()
    built_few_shot_prompts = []
    pattern_col = '(f_col\(\[(.*?)\]\))'
    pattern_col = re.compile(pattern_col, re.S)

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
            # pred = list(set().union(*preds,col_dict[str(g_eid)]['output']))
            all_preds = preds + [col_dict[str(g_eid)]["output"]]
            print(f'过滤后得preds是{preds}，col_dict[str(g_eid)]["output"]是{[col_dict[str(g_eid)]["output"]]}')
            pred = set().union(*all_preds)
            print(f"pred是{pred}")
            
            
            
            #对前两步(col_text和col_sql)找到的关键列名进行去重
            g_data_item = col_dict[str(g_eid)]['data_item']
            g_dict[g_eid] = {
                'generations': [],
                'cols' : [],
                'ori_data_item': copy.deepcopy(g_data_item)
            }
            db = NeuralDB(
                tables=[{'title': g_data_item['table']['page_title'], 'table': g_data_item['table']}]
            )
            #表格入库的时候为每张表增加了 row_id 这一列
            g_data_item['table'] = db.get_table_df()
            g_data_item['title'] = db.get_table_title()

            filtered_pred = [value for value in g_data_item['table'].columns if value in pred]#预测出的列一定要是在表头里有
            
            
            #如果预测出的列在原表完全没有出现，那就对整张表进行问答，不过滤
            if filtered_pred == []:
                filtered_pred = [value for value in g_data_item['table'].columns]
                
            g_dict[g_eid]['cols'] = filtered_pred#cols是关键的列

            #开始为过滤的子表做准备-->构造子表的表头：如果 filtered_pred 中没有 'row_id' 这个列名，就把它加到最前面
            if 'row_id' not in filtered_pred:
                filtered_pred.insert(0, 'row_id')

            df = g_data_item['table'][filtered_pred]#这个时候df里面 row_id是存在的
            
            #现在准备输入给大模型的表g_data_item['table'] 已经是过滤后的子表了
            g_data_item['table'] = df

            n_shots = args.n_shots
            
            few_shot_prompt = generator.build_few_shot_prompt_from_file(
                file_path=args.prompt_file,
                n_shots=n_shots
            )
            print(f'shot模板构建完成！！！！！！！！！！！！！！！！')
            
            generate_prompt = generator.build_generate_prompt(
                data_item=g_data_item,
                generate_type=(args.generate_type,)
            )
            print(f'生成模板构建完成！！！！！！！！！！！！！！！！')
            prompt = few_shot_prompt + "\n\n" + generate_prompt
            # Ensure the input length fit max input tokens by shrinking the number of rows
            max_prompt_tokens = args.max_api_total_tokens - args.max_generation_tokens
            num_rows = (g_data_item['table'].shape[0])

            print(f'行数是{num_rows}！！！！！！！！！！！！！！！！')

            while len(tokenizer.encode(prompt)) >= max_prompt_tokens:
                num_rows = 5
                generate_prompt = generator.build_generate_prompt(
                data_item=g_data_item,
                generate_type=(args.generate_type,),
                num_rows = num_rows
                )
                prompt = few_shot_prompt + "\n\n" + generate_prompt
            # print(generate_prompt)#注意 
            print(f"Process#{pid}: Building prompt for eid#{g_eid}, original_id#{g_data_item['id']}")
            built_few_shot_prompts.append((g_eid, prompt))

            if len(built_few_shot_prompts) < args.n_parallel_prompts:
                print("草泥马 提示数量不够啊")
                continue

            print(f"Process#{pid}: Prompts ready with {len(built_few_shot_prompts)} parallels. Run openai API.")
            response_dict = generator.generate_one_pass(
                prompts=built_few_shot_prompts,
                verbose=args.verbose
            )
            
            
            
            # Run the generated SQL on an interpreter to get rows
            #g_pairs[n] 是最后用生成的SQL抽取到的相关的行，用exec_rows保存
            for eid, g_pairs in response_dict.items():
                
                try:
                    g_dict[eid]['generations'] = g_pairs
                    print(f"第{eid}个结果是{g_pairs}")
                except NameError:
                # 如果 g_pairs 未定义时的处理
                    print("Error: g_pairs is not defined.")
                

                

                for idx, n in enumerate(range(len(g_pairs))):
                    print(f'这是第{idx+1}次到这里')
                    
                    
                    exec_rows = []
                    
                    keys = ""
                    try:
                        executor = Executor(args, keys)
                        sql = g_pairs[n].split('SQL: ')[1]
                        try:
                            sql
                            print(f'SQL的内容是{sql}')
                            
                        except NameError:
                            print("变量不存在！")

                        norm_sql = post_process_sql(
                        sql_str=sql,
                        df=db.get_table_df(),
                        process_program_with_fuzzy_match_on_db=True,
                        table_title= g_data_item['title']
                        )
                        #执行sql查询：exec_answer
                        #(找到)['John', 'USA', '2020', 'Mary', 'UK', '2021']这种的单个样本

                        exec_answer = executor.sql_exec(norm_sql, db, verbose=False)
                        
                        #exec_rows返回的是['row 1', 'row 2', 'row 3']这样的
                        exec_rows = extract_rows(exec_answer)
                        if  exec_rows:
                            print(f'成功提取{exec_rows}')
                        else :
                            print('抽取失败 extract_rows没有得到相应得rows信息')
                        g_pairs[n] += f' f_row({str(exec_rows)})'
                        
                        
                    except:
                        g_pairs[n] += f' f_row({str(exec_rows)})'
         
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
    col_dict = dict()
    for eid, _ in data.items():
        data_item = data[eid]['ori_data_item']
        col_output = data[eid]['cols']
        if data[eid]['generations'] :
            col_gen = data[eid]['generations']
        else:   
            col_gen = []

        col_dict[eid] = {'cols': col_gen, 'data_item':data_item, 'output': col_output}
    # Split by processes
    col_dict_group = [dict() for _ in range(args.n_processes)]
    # for idx, eid in enumerate(col_dict.keys()):
    #     col_dict_group[idx % args.n_processes][eid] = col_dict[eid]
    for idx, eid in enumerate(sorted(col_dict.keys(), key=int)):  # 按整数排序
        col_dict_group[idx % args.n_processes][eid] = col_dict[eid]

    print(len(col_dict))
    # Annotate
    generator = Generator(args, keys=keys)
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
    save_file_name = f'row_sql_1.json'
    with open(os.path.join(args.save_dir, save_file_name), 'w') as f:
        json.dump(g_dict, f, indent=4)

    print(f"Elapsed time: {time.time() - start_time}")
    



if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--api_keys_file", type=str, default="key.txt")
    parser.add_argument("--prompt_file", type=str, default="./prompts/row_select_sql.txt")

    parser.add_argument(
        "--generate_type",
        type=str,
        default="row",
        choices=["col", "answer", "row", "verification"],
    )
    parser.add_argument("--n_shots", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    # LLM options
    parser.add_argument(
        "--engine", type=str, default="Qwen/Qwen2.5-7B-Instruct"
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
    # Multiprocess options
    parser.add_argument("--n_processes", type=int, default=5)
    parser.add_argument("--input_program_file", type=str, default="col_text.json")

    # 先和上一层一样 先加载原始的表格数据和API KEYS=========================================================
    args = parser.parse_args()
    args.dataset_path = f"./datasets/data/fetaQA-v1_test.json"
    
    main()