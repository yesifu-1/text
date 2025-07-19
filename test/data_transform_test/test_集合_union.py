
preds=["Album",
      "Single",
      "Year"]
col_dict={}
col_dict['output']=['let_go','dada']
pred = list(set().union(*preds,col_dict['output']))
print(pred)