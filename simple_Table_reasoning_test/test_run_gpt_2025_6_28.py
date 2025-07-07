import os
import time
import json
import pandas as pd
from transformers import AutoTokenizer
import tiktoken
import argparse
from generation.generator_gpt_2 import Generator  # 核心生成器类
import copy
parser = argparse.ArgumentParser()


parser.add_argument(
    "--dataset", type=str, default="wikitq", choices=["wikitq", "tab_fact", "fetaqa"]
)
parser.add_argument(
    "--dataset_split", type=str, default="test", choices=["train", "validation", "test"]
)
parser.add_argument("--api_keys_file", type=str, default="key.txt")
parser.add_argument("--prompt_file", type=str, default="./prompts/col_select_sql.txt")
parser.add_argument("--save_dir", type=str, default="results/model_gpt")

# Multiprocess options
parser.add_argument("--n_processes", type=int, default=1)

# Program generation options
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

args = parser.parse_args()

args.dataset_path = f"./datasets/data/fetaQA-v1_test.json"


# ==== 加载表格数据（JSON） ====
with open(args.dataset_path, "r") as f:
    dataset = json.load(f)

# ==== 加载 API keys ====
with open(args.api_keys_file, "r") as f:
    keys = [line.strip() for line in f if line.strip()]

# ==== 初始化 Generator 和 Tokenizer ====
generator = Generator(args, keys=keys)

tokenizer = tiktoken.get_encoding("cl100k_base")
if generator and tokenizer:
    print(f"✓ 成功初始化生成器和分词器")
else:
    print("❌ 生成器或分词器初始化失败，请检查配置")


# ==== 选择一个样本进行测试 ====
eid = 0
g_data_item = dataset["data"][eid]
print(f"[Testing eid={eid}] id={g_data_item['feta_id']}")

#================================================
#保存原始数据副本
# ori_data_item=copy.deepcopy(g_data_item)

#================================================


table_array = g_data_item["table_array"]

if len(table_array) < 2:
    raise ValueError("表格数据不足，无法构建 DataFrame")#不满足一行列名+一行数据


# 如果第二行和第一行之间高度相似（即表头结构多余），就选第二行为列名
def is_header_row(row):
    return all(isinstance(cell, str) for cell in row)


if (
    len(table_array) >= 3
    and is_header_row(table_array[0])
    and is_header_row(table_array[1])
):
    header = table_array[1]
    rows = table_array[2:]
else:
    header = table_array[0]
    rows = table_array[1:]


df = pd.DataFrame(rows, columns=header)#把列表形式表示的表格数据转换为 DataFrame
data_to_llm=copy.deepcopy(g_data_item)#深拷贝以后，原始的表格可以任意修改，不会影响传入给llm的表格数据
test_data_to_result=copy.deepcopy(g_data_item)
data_to_result= {
                "id": test_data_to_result['feta_id'],
                "table": {
                    "id": test_data_to_result['feta_id'],
                    "header": header,
                    "rows": rows,
                    "page_title":  test_data_to_result['table_page_title'],
                },
                "question": test_data_to_result['question'],
                "answer": test_data_to_result['answer']
                }
# data_to_result=
breakpoint()
data_to_llm["table"] = df

g_data_item["table"] = df


# 设置表格标题字段（如果有）
data_to_llm["table_page_title"] = data_to_llm.get("table_page_title", "Untitled Table")

# ==== 构造 prompt ====
few_shot_prompt = generator.build_few_shot_prompt_from_file(
    file_path=args.prompt_file, n_shots=args.n_shots
)
generate_prompt = generator.build_generate_prompt(#针对要进行问答的表格构建提示
    data_item=data_to_llm, generate_type=(args.generate_type,)
)
prompt = few_shot_prompt + "\n\n" + generate_prompt

# ==== 控制 token 长度（如过长，裁剪表格行数）====
max_prompt_tokens = args.max_api_total_tokens - args.max_generation_tokens

if len(tokenizer.encode(prompt)) >= max_prompt_tokens:
    generate_prompt = generator.build_generate_prompt(
        data_item=g_data_item,
        generate_type=(args.generate_type,),
        num_rows=5,  # 裁剪为5行
    )
    prompt = few_shot_prompt + "\n\n" + generate_prompt


# ==== 调用大模型 ====

print(">> 调用大模型生成中...")
start_time = time.time()
response_dict = generator.generate_one_pass(
    prompts=[(eid, prompt),(eid+1,prompt)], verbose=args.verbose
)
print(f">> 耗时：{time.time() - start_time:.2f} 秒")

# ==== 打印/保存结果 ====
response=[]

for (i,res) in response_dict.items():
    response.extend(res)

g_dict=dict()
g_dict[eid] = set()
g_dict[eid]={'ori_data_item':data_to_result,'generations':response}

for i, text in enumerate(response):
    print(f"\n[回答 {i + 1}] : {text}\n")

# 可选保存
os.makedirs(args.save_dir, exist_ok=True)
with open(
    os.path.join(
        args.save_dir, f"{args.dataset}_{args.dataset_split}_test_result.json"
    ),
    "w",
) as f:
    json.dump(g_dict, f, indent=2)

print(">> 测试完成")
