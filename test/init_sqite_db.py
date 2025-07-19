import sqlite3
import pandas as pd

# 连接 SQLite 数据库
conn = sqlite3.connect('database.db')
# 读取 CSV 文件
df = pd.read_csv(r'D:\\project\\gittest.text\\TbaleReasoningWithLLM\\test\\tables_csv\\6_GE U28C.csv')
# 导入到 SQLite 表
df.to_sql('table_name', conn, if_exists='replace', index=False)