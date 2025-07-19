import pandas as pd


table_header=['name','city']
table_rows=[['jch','kaizhou'],['wxf','chongqing']]

#从json的某一个列表转换为df表格
df = pd.DataFrame(table_rows,columns=table_header)
print(df)
# print(f"{df.columns.tolist()} {df.values.tolist()}")
df=df.T
# print(df)
string = '/*\n'   
string+= 'row : '

# Add header row with row numbers (limited to 15 rows)
for idx,i in enumerate(df.columns.tolist()):
    if idx == 15 or idx == len(df.columns):
        break
    string += 'row {}'.format(i)
    if idx != 14 and idx != len(df.columns)-1:
        string += ' | '
string += '\n'
for row_id, row in df.iloc[:14].iterrows():
    string += f'{row_id} : '
    for column_id, header in enumerate(df.columns):
        string += str(row[header])
        if column_id == 14 or column_id == len(df.columns)-1:
            break
        if column_id != 15 or column_id != len(df.columns)-1:
            string += ' | '
    string += '\n'
string += '*/\n'
string += f'columns:{table_header}\n'
print(string)

