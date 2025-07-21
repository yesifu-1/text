import os
import time
start_time=time.time()
os.system(fr"""python muiti_process_script/multi_col_sql.py --engine --n_processes 5  --save_dir results/multi_process_model_gpt """)

os.system(fr"""python muiti_process_script/multi_col_text.py --engine --n_processes 5  --save_dir results/multi_process_model_gpt --input_program_file col_sql.json """)

os.system(fr"""python muiti_process_script/multi_row_sql.py --n_processes  5 --save_dir results/multi_process_model_gpt --input_program_file col_text.json""")

os.system(fr"""python muiti_process_script/multi_row_text.py --n_processes 5 --save_dir results/multi_process_model_gpt --input_program_file row_sql.json """)

os.system(fr"""python muiti_process_script/multi_sql_reason.py --n_processes 5 --save_dir results/multi_process_model_gpt --input_program_file row_text.json """)

os.system(fr"""python muiti_process_script/multi_text_reason.py --n_processes 5  --save_dir results/multi_process_model_gpt --input_program_file reason_sql.json """)

end_time=time.time()
print(f"处理单张表格耗时{end_time-start_time:.2f}s")
