#注意 这一步的generate_type是verification，为什么？？？
# 注意 实际上除了第一步col_sql 其他时候都不用重新读取数据集
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
parser.add_argument("--prompt_file", type=str, default="./prompts/row_select_text.txt")

parser.add_argument(
    "--generate_type",
    type=str,
    default="verification",
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
parser.add_argument("--input_program_file", type=str, default="fetaqa_row_sql.json")

# 先和上一层一样 先加载原始的表格数据和API KEYS=========================================================
args = parser.parse_args()
args.dataset_path = f"./datasets/data/fetaQA-v1_test.json"

with open(args.api_keys_file, "r") as f:
    keys = [line.strip() for line in f if line.strip()]

generator = Generator(args, keys=keys)

tokenizer = tiktoken.get_encoding("cl100k_base")
if generator and tokenizer:
    print(f"✓ 成功初始化生成器和分词器")
else:
    print("❌ 生成器或分词器初始化失败，请检查配置")

with open(os.path.join(args.save_dir, args.input_program_file), "r") as f:
    data = json.load(f)
    row_dict = dict()
    for eid, _ in data.items():
        data_item = data[eid]['ori_data_item']
        if data[eid]['generations'] or data[eid]['cols']:
            rows = data[eid]['generations']
            cols = data[eid]['cols']
        else:
            rows = []
            cols = []
        row_dict[eid] = {'rows': rows, 'cols': cols, 'data_item' : data_item}
        #rows是准备清洗的上一阶段的gen信息，cols是之前已经清洗好的，data_item是原始数据

#============进入处理逻辑========================
g_eid = 0  # 注意 手动test
g_dict = dict()
built_few_shot_prompts = []
pattern_row = '(f_row\(\[(.*?)\]\))'
pattern_row_num = '\d+'
pattern_row = re.compile(pattern_row, re.S)
pattern_row_num = re.compile(pattern_row_num, re.S)

preds = []
n = 0

try:
    pred = re.findall(pattern_row,row_dict[str(g_eid)]['rows'][n])[0][1]
    if pred == '*':
        pred = ''
        for i in range(len(row_dict[str(g_eid)]['data_item']['table']['rows'])):
            pred += f'row {i}'
            if i != len(row_dict[str(g_eid)]['data_item']['table']['rows'])-1:
                pred += ', '
    pred = pred.replace("'","")
    pred = pred.split(', ')
    preds.append(pred)
    
            

except:
    pass
pred = list(set().union(*preds))
g_data_item = row_dict[str(g_eid)]['data_item']
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
filtered_cols = [value for value in g_data_item['table'].columns if value in row_dict[str(g_eid)]['cols']]
if filtered_cols == []:
    filtered_cols = [value for value in g_data_item['table'].columns]
g_dict[g_eid]['cols'] = filtered_cols

if [row_dict[str(g_eid)]['cols']] != []:
    df = g_data_item['table'][filtered_cols]

g_dict[g_eid]['rows'] = pred
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

prompt1, prompt2 = generate_prompt.split('<initial response>')
prompt = few_shot_prompt + "\n\n" + prompt1 + f'Initial Response:{pred}' + "\n" + prompt2

print(generate_prompt)

max_prompt_tokens = args.max_api_total_tokens - args.max_generation_tokens
#=========和之前一样，控制token 显示有限的行====================================================
num_rows = (g_data_item['table'].shape[0])

while len(tokenizer.encode(prompt)) >= max_prompt_tokens:
    num_rows = 5
    generate_prompt = generator.build_generate_prompt(
    data_item=g_data_item,
    generate_type=(args.generate_type,),
    num_rows = num_rows
    )
    prompt = few_shot_prompt + "\n\n" + prompt1 + f'Rows:{pred}' + "\n" + prompt2
print(f"Process# pid0: Building prompt for eid#{g_eid}, original_id#{g_data_item['id']}")

built_few_shot_prompts.append((g_eid, prompt))

# =================调用模型进行相关行抽取========================================
start_time = time.time()
response_dict = generator.generate_one_pass(
                prompts=built_few_shot_prompts,
                verbose=args.verbose
            )
print(f">> 耗时：{time.time() - start_time:.2f} 秒")
for i, res in response_dict.items():
    print(f"第{i}个行抽取回答的内容是-------------> {res}")

 #存入generation   
for eid, g_pairs in response_dict.items():

    g_dict[int(eid)]["generations"] = g_pairs

# ==========================保存结果=============================================
save_file_name = f"{args.dataset}_row_text.json"
os.makedirs(args.save_dir, exist_ok=True)

with open(os.path.join(args.save_dir, save_file_name),"w") as f:
    json.dump(g_dict, f, indent=2)


print(">> 测试完成")
