## 表格问答主要6步:col_sql,col_text ,row_sql,row_text , reason_sql,reason_text
### col_sql和col_text : 分别生成sql语句和直接基于text进行关键列过滤。两个步骤选出来的关键列要进行去重。
### row_sql和row_text : 基于前2步选出的关键列，row_sql生成sql进行关键行筛选，把row_sql的结果解析后添加到row_text部分提示模板的<initial response>中，再次进行过滤。两个步骤选出来的关键行要进行去重。
### reason_sql和reason_text:用之前四个步骤检索回的内容输出一个回答原始问题的SQL语句，筛选出关键的行得到最终用于问答的Additional Evidence，也就是过滤后的子表。在reason_text部分进行最终问答。
