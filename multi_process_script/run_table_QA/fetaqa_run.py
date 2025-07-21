import os
import time
start_time=time.time()
# os.system(fr"""python ./multi_process_script/multi_col_sql.py --engine Qwen/Qwen2.5-32B-Instruct --n_processes 5  --save_dir results/multi_process_model_gpt """)

# os.system(fr"""python ./multi_process_script/multi_col_text.py --engine Qwen/Qwen2.5-32B-Instruct --n_processes 5  --save_dir results/multi_process_model_gpt --input_program_file col_sql.json """)

# os.system(fr"""python ./multi_process_script/multi_row_sql.py --n_processes 5 --engine Qwen/Qwen2.5-32B-Instruct --save_dir results/multi_process_model_gpt --input_program_file col_text.json""")

# os.system(fr"""python ./multi_process_script/multi_row_text.py --n_processes 5 --engine Qwen/Qwen2.5-32B-Instruct --save_dir results/multi_process_model_gpt --input_program_file row_sql.json """)

# os.system(fr"""python ./multi_process_script/multi_sql_reason.py --n_processes 5 --engine Qwen/Qwen2.5-32B-Instruct --save_dir results/multi_process_model_gpt --input_program_file row_text.json """)

os.system(fr"""python ./multi_process_script/multi_text_reason.py --n_processes 5  --engine Qwen/Qwen2.5-32B-Instruct --save_dir results/multi_process_model_gpt --input_program_file reason_sql.json --prompt_file ./prompts/fetaqa.txt""")

end_time=time.time()
print(f"处理15张耗时{end_time-start_time:.2f}s")
