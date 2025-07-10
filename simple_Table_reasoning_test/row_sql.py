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
from utils.utils import load_data_split
from nsql.database import NeuralDB
from nsql.parser import extract_rows
from utils.normalizer import post_process_sql
from nsql.sql_exec import Executor, extract_rows

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
parser.add_argument("--save_dir", type=str, default="results/model_gpt")
parser.add_argument("--input_program_file", type=str, default="fetaqa_col_text.json")

# 先和上一层一样 先加载原始的表格数据和API KEYS=========================================================
args = parser.parse_args()
args.dataset_path = f"./datasets/data/fetaQA-v1_test.json"

# dataset=[]
# with open(args.dataset_path, "r") as f:
#     dataset = json.load(f)
#     dic=dataset["data"][0]
#     breakpoint()

#     # print(dic)
#     feta_id = dic['feta_id']
#     caption = dic['table_page_title']
#     question = dic['question']
#     sub_title = dic['table_section_title']
#     answer = dic["answer"]
#     header = dic['table_array'][0]
#     rows = dic['table_array'][1:]
#     data = {
#     "id": feta_id,
#     "table": {
#         "id": feta_id,
#         "header": header,
#         "rows": rows,
#         "page_title": caption,
#         "sub_title": sub_title
#     },
#     "question": question,
#     "answer": answer
#     }
#     breakpoint()   #在提取行这一步不需要读取原始数据集

with open(args.api_keys_file, "r") as f:
    keys = [line.strip() for line in f if line.strip()]

generator = Generator(args, keys=keys)


tokenizer = tiktoken.get_encoding("cl100k_base")
if generator and tokenizer:
    print(f"✓ 成功初始化生成器和分词器")
else:
    print("❌ 生成器或分词器初始化失败，请检查配置")


# 读取前两步结果，很重要
with open(os.path.join(args.save_dir, args.input_program_file), "r") as f:
    data = json.load(f)
    col_dict = dict()
    for eid, _ in data.items():
        data_item = data[eid]["ori_data_item"]
        col_output = data[eid]["cols"]
        if data[eid]["generations"]:
            col_gen = data[eid]["generations"]
        else:
            col_gen = []

        col_dict[eid] = {"cols": col_gen, "data_item": data_item, "output": col_output}
        # col_dict[eid]是要输入给正式处理逻辑的内容(包括了第二阶段的生成答案cols，第一阶段的预测列output以及原始信息data_item)，下一步我们需要将其中col_text预测中的结果'cols': col_gen和之前步骤col_sql的结果data_item进行去重。


g_eid = 0  # 注意 手动test
g_dict = dict()
built_few_shot_prompts = []
pattern_col = "(f_col\(\[(.*?)\]\))"
pattern_col = re.compile(pattern_col, re.S)

preds = []
# for n in range(args.sampling_n): #这里测试 只采样一次 所以直接n写成0
n = 0
try:
    pred = re.findall(pattern_col, col_dict[str(g_eid)]["cols"][n])[0][1]
    pred = pred.split(", ")
    new_pred = []
    for i in pred:
        if i.startswith("'"):
            i = i[1:]

        if i.endswith("'"):
            i = i[:-1]
        new_pred.append(i)
    print(
        f"第二步col_text预测出的列是{new_pred}，下一步是把这些列和第一步中col_sql的结果进行去重合并"
    )
    preds.append(new_pred)  # preds形如[['Year', 'Single', 'Album'],['Single', 'Album']]

except:
    pass
# for strs in pred:
#     pred.lower()


# pred = list(set().union(*preds,col_dict[str(g_eid)]['output']))
all_preds = preds + [col_dict[str(g_eid)]["output"]]
pred = set().union(*all_preds)

# 注意:手动处理 把前两步预测的列全部变成小写
pred = [big.lower() for big in pred]


# 对前两步(col_text和col_sql)找到的关键列名进行去重
g_data_item = col_dict[str(g_eid)]["data_item"]
g_dict[g_eid] = {
    "generations": [],
    "cols": [],
    "ori_data_item": copy.deepcopy(g_data_item),
}
db = NeuralDB(
    tables=[
        {"title": g_data_item["table"]["page_title"], "table": g_data_item["table"]}
    ]
)
# 表格入库的时候为每张表增加了 row_id 这一列
g_data_item["table"] = db.get_table_df()
g_data_item["title"] = db.get_table_title()

# df.columns.tolist()
filtered_pred = [
    value for value in g_data_item["table"].columns.tolist() if value in pred
]  # 保证预测出的列一定要是在表头里有

# 如果预测出的列在原表完全没有出现，那就对整张表进行问答，不过滤
if filtered_pred == []:
    filtered_pred = [value for value in g_data_item["table"].columns]
g_dict[g_eid]["cols"] = filtered_pred  # cols是关键的列

# 开始为过滤的子表做准备-->构造子表的表头：如果 filtered_pred 中没有 'row_id' 这个列名，就把它加到最前面
if "row_id" not in filtered_pred:
    filtered_pred.insert(0, "row_id")

df = g_data_item["table"][filtered_pred]  # 这个时候df里面 row_id是存在的

# 现在准备输入给大模型的表g_data_item['table'] 已经是过滤后的子表了
g_data_item["table"] = df

n_shots = args.n_shots
few_shot_prompt = generator.build_few_shot_prompt_from_file(
    file_path=args.prompt_file, n_shots=n_shots
)
generate_prompt = generator.build_generate_prompt(
    data_item=g_data_item, generate_type=(args.generate_type,)
)

prompt = few_shot_prompt + "\n\n" + generate_prompt

# Ensure the input length fit max input tokens by shrinking the number of rows
max_prompt_tokens = args.max_api_total_tokens - args.max_generation_tokens
num_rows = g_data_item["table"].shape[0]

while len(tokenizer.encode(prompt)) >= max_prompt_tokens:
    num_rows = 5
    generate_prompt = generator.build_generate_prompt(
        data_item=g_data_item, generate_type=(args.generate_type,), num_rows=num_rows
    )
    prompt = few_shot_prompt + "\n\n" + generate_prompt
print(generate_prompt)
print(
    f"正在处理进程1: Building prompt for eid#{g_eid}, original_id#{g_data_item['id']}"
)
built_few_shot_prompts.append((g_eid, prompt))


# =================调用模型进行相关行抽取========================================
start_time = time.time()
response_dict = generator.generate_one_pass(
    prompts=[(eid, prompt), (str(int(eid) + 1), prompt)], verbose=args.verbose
)
print(f">> 耗时：{time.time() - start_time:.2f} 秒")
for i, res in response_dict.items():
    print(f"第{i}个行抽取回答的内容是-------------> {res}")


# ===================Run the generated SQL on an interpreter to get rows==========
# =================== #g_pairs[n] 是最后用生成的SQL抽取到的相关的行，放到generations的末尾=======
for eid, g_pairs in response_dict.items():

    g_dict[int(eid)]["generations"] = g_pairs

    n = 0
    exec_rows = []
    keys = ""
    try:
        executor = Executor(args, keys)
        sql = g_pairs[n].split("SQL: ")[1]
        norm_sql = post_process_sql(
            sql_str=sql,
            df=db.get_table_df(),
            process_program_with_fuzzy_match_on_db=True,
            table_title=g_data_item["title"],
        )
        # 执行sql查询：exec_answer
        # (找到)['John', 'USA', '2020', 'Mary', 'UK', '2021']这种的单个样本

        exec_answer = executor.sql_exec(norm_sql, db, verbose=False)

        # exec_rows返回的是['row 1', 'row 2', 'row 3']这样的

        exec_rows = extract_rows(exec_answer)

        g_pairs[
            n
        ] += f" f_row({str(exec_rows)})"  # 和g_dict[int(eid)]['generations']是共享内存
        

        if eid == "0":
            break  # 注意，手动跳出，因为只有一个样本
    except:
        g_pairs[n] += f" f_row({str(exec_rows)})"  # 如果中间解析不对，就不进行处理


# ==========================保存结果=============================================
save_file_name = f"{args.dataset}_row_sql.json"
os.makedirs(args.save_dir, exist_ok=True)

with open(os.path.join(args.save_dir, save_file_name),"w") as f:
    json.dump(g_dict, f, indent=2)


print(">> 测试完成")
