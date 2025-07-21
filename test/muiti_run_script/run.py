import os
import time
start_time=time.time()
os.system(fr"""python ./simple_Table_reasoning_test/gpt_script/col_sql.py """)
os.system(fr"""python ./simple_Table_reasoning_test/gpt_script/col_text.py """)
os.system(fr"""python ./simple_Table_reasoning_test/gpt_script/row_sql.py """)
os.system(fr"""python ./simple_Table_reasoning_test/gpt_script/row_text.py """)
os.system(fr"""python ./simple_Table_reasoning_test/gpt_script/reason_sql.py """)
os.system(fr"""python ./simple_Table_reasoning_test/gpt_script/reason_text.py """)
end_time=time.time()
print(f"处理单张表格耗时{end_time-start_time:.2f}s")
