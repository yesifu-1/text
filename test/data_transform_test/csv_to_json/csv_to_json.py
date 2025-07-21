import csv
import json
import os
from pathlib import Path

def read_tsv_file(tsv_path):
    """读取 TSV 文件，返回样本列表"""
    samples = []
    try:
        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 4:  # 确保包含 id, utterance, context, targetValue
                    samples.append({
                        'id': row[0],
                        'question': row[1],
                        'table_path': row[2],
                        'answer': row[3]
                    })
    except FileNotFoundError:
        print(f"TSV file not found: {tsv_path}")
        return []
    except Exception as e:
        print(f"Error reading TSV file {tsv_path}: {e}")
        return []
    return samples

def read_csv_table(csv_path):
    """读取 CSV 文件，返回表格数组"""
    table_array = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                table_array.append(row)
    except FileNotFoundError:
        print(f"CSV file not found: {csv_path}")
        return []
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return []
    return table_array

def convert_to_json(tsv_files, dataset_dir, output_json_path):
    """将 TSV 和 CSV 文件转换为指定 JSON 格式"""
    output_data = {
        "version": "v1",
        "split": "dev",
        "data": []
    }

    for tsv_file in tsv_files:
        tsv_path = os.path.join(dataset_dir, 'data', tsv_file)
        samples = read_tsv_file(tsv_path)
        
        for sample in samples:
            # 读取对应的 CSV 表格
            csv_path = os.path.join(dataset_dir, sample['table_path'])
            table_array = read_csv_table(csv_path)
            
            if not table_array:
                print(f"Skipping sample {sample['id']} due to missing or invalid table")
                continue

            # 处理 targetValue，可能包含 | 分隔的多值
            answer = sample['answer']
            if '|' in answer:
                answer = answer.replace('|', ', ')  # 转换为逗号分隔的字符串

            # 构建 JSON 条目
            json_entry = {
                "feta_id": sample['id'],
                "table_source_json": sample['table_path'],
                "page_wikipedia_url": "http://en.wikipedia.org/wiki/Unknown",
                "table_page_title": "Unknown",
                "table_section_title": "Unknown",
                "table_array": table_array,
                "highlighted_cell_ids": [],  # WikiTableQuestions 无高亮信息
                "question": sample['question'],
                "answer": answer,
                "source": "wikitablequestions"
            }
            output_data["data"].append(json_entry)

    # 保存 JSON 文件
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        print(f"Output saved to {output_json_path}")
    except Exception as e:
        print(f"Error writing JSON file {output_json_path}: {e}")

def main():
    # 配置
    dataset_dir = r"./datasets/wikitableQA"  # 替换为你的数据集路径
    output_json_path = r"./test/data_transform_test/csv_to_json/converted_dev.json"
    tsv_files = ["random-split-1-dev.tsv"]

    # 转换
    convert_to_json(tsv_files, dataset_dir, output_json_path)

if __name__ == "__main__":
    main()