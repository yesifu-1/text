#1,在提示中包含表格
# string='create table test_table \n'
# headers=['name','age','city']
# for head in headers:

#     column_type=str(type(head)).replace('<class',' ').replace('>',' ')
    
#     string += '\t{} {},\n'.format(head, column_type)

# print(string)   


#2，测试df.iloc[:num_rows].iterrows()  
#性能较慢，不推荐用于大数据量处理；如果只是遍历值，itertuples() 更快。

import pandas as pd

df = pd.DataFrame({
    'name': ['Alice', 'Bob'],
    'age': [20, 22]
})
print(df.columns)#这是Index类，类似列表，说明了这个df的列明
print(df.columns.values)#ndarray列表 ['Alice', 'Bob']
print(df.columns.values.tolist()) #list列表

   
# for idx,row in df.iloc[:2].iterrows():
#     print(f"id:{idx},name:{row['name']}")
