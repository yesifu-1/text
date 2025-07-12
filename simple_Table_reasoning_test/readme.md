
## 第一步
### col_sql的输出:生成sql对表格关键列筛选后,包含Evidence: f_col([列名1，列名2])的格式的回复(generations)

## 第二步
### col_text的输出 :用text对表格关键列筛选后Evidence: f_col([列名1, 列名2, 列名n])" (generations)和cols (col_sql中山选出的列 ，在这一层被解析为了列表，类似于 "cols":["Album"，"Single","Year"],)

## 第三步
### row_sql的输出 :包含类似"SQL: SELECT single FROM w WHERE album = 'across the rio grande' AND year = '1988'; f_row(['row 7', 'row 8'])"的字符串，输出中的cols字段会包含一个列表类似"cols":[ "row_id","Album"，"Single","Year"],  row_id为我们手动添加 。这一步中我们解析了来自上一步的generation中的字符串，解析关键列后与上一列的cols(第一步选出的重要列)去重，合并，全部放在了cols字段中。

## 第四步 解析前面的拿到的 包含类似(['row 7', 'row 8'])"的字符串，用text重新进行问答:再次拿到属于本层的 包含类似(['row 7', 'row 8'])"的字符串,

## 第五步 猜测:解析row_text步骤的字符串，从gen字符串提取出['row 7', 'row 8']，