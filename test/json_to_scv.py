import json
import csv
import os
import re

# 读取 JSON 文件
with open('fetaQA-v1_test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 创建文件夹
output_folder = 'tables_csv'
os.makedirs(output_folder, exist_ok=True)

# 遍历 JSON 中的每个表数据
for item in data['data']:
    table_id = item['feta_id']
    table_title = re.sub(r'[:"*?<>|]', '_', item['table_page_title'])  # 替换无效字符
    table_data = item['table_array']

    # 生成 CSV 文件名
    csv_filename = f"{output_folder}/{table_id}_{table_title}.csv"

    # 写入 CSV 文件
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(table_data)

print(f"CSV files generated in {output_folder}")