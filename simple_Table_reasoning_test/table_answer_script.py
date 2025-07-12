import os
import time
start_time=time.time()
os.system(fr"""python ./simple_Table_reasoning_test/col_sql.py """)
os.system(fr"""python ./simple_Table_reasoning_test/col_text.py """)
os.system(fr"""python ./simple_Table_reasoning_test/row_sql.py """)
os.system(fr"""python ./simple_Table_reasoning_test/row_text.py """)
os.system(fr"""python ./simple_Table_reasoning_test/reason_sql.py """)
os.system(fr"""python ./simple_Table_reasoning_test/reason_text.py """)
end_time=time.time()
print(f"处理单张表格耗时{end_time-start_time:.2f}s")
