import json
from evaluate import load
# from utils.evaluator import Evaluator,EvaluateTool
output=dict()
with open (r'D:/project/gittest.text/TbaleReasoningWithLLM/results/model_gpt/fetaqa_reason_text.json','r') as f:
    output = json.load(f)
    
    
gold_answer = [output['0']['ori_data_item']["answer"]]
print(gold_answer)
pred_answer=[output['0']['generations'][0].split('the answer is: ')[1]]
print(pred_answer)
# dataset='wikitq'
# args=[]
print("" "")
rouge=load('rouge')

# score = load(
#         pred_answer,
#         gold_answer,)
rouge_types=['rouge1'] #用rouge_types来规定算哪几个指标不然就默认返回全部四个
result= rouge.compute(predictions=pred_answer,references=gold_answer, use_aggregator=False,rouge_types=rouge_types)
print(result)