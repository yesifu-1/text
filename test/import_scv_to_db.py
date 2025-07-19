import os
import pandas as pd
import sqlite3

conn = sqlite3.connect('database.db')
folder_path = 'tables_csv'
for file in os.listdir(folder_path):
    if file.endswith('.csv'):
        table_name = os.path.splitext(file)[0]  # 使用文件名作为表名
        df = pd.read_csv(os.path.join(folder_path, file))
        df.to_sql(table_name, conn, if_exists='replace', index=False)
conn.close()