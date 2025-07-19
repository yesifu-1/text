import pandas as pd
import sqlite3
conn = sqlite3.connect('database.db')
cursor = conn.cursor()
results = cursor.execute("SELECT * FROM `43_Richard Thompson (sprinter)` LIMIT 10")#cursor和results都是sqlite3.Cursor对象

#因为表名包含特殊字符和空格，需要用反引号或方括号包围表名
# 所以以下代码是错误的
# results =cursor.execute("SELECT * FROM 43_Richard Thompson (sprinter) LIMIT 10")

#查询结果不包括表头，所以在查看结果时需要手动添加列名(如果想以子表的形式查看结果)
# columns = [desc[0] for desc in cursor.description]  # 获取列名
# df = pd.DataFrame(results, columns=columns)
#这里的results既可以直接写cursor对象也可以是fetch后得到的列表
# print(df)


# print(type(cursor.fetchall()))
conn.close()