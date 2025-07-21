import json
from evaluate import load
import time
# from utils.evaluator import Evaluator,EvaluateTool
output=dict()
with open (r'D:/project/gittest.text/TbaleReasoningWithLLM/test/evaluate/fetaqa_Qwen2.5-32B-Instruct_2025_7_21_2243.json','r') as f:
    output = json.load(f)
    end=[]
    all=[]
    start=time.time()
    out_put_dict=dict()#保存每一个标准答案，生成的回答，回答的分数，放到一个json用于对比
    idx=0
    for key in output:
        idx+=1
        ori_question=output[key]['ori_data_item']['question']
        print(f'第{idx}个问题是: \n {ori_question}')
    
        gold_answer = [output[key]['ori_data_item']["answer"]]
        print(f'标准答案是{gold_answer}')
        try:
            pred_answer=[output[key]['generations'][0].split('the answer is: ')[1]]#遇到没有遵照模板的情况，会split失败,跳过即可
        except:
            pred_answer=[]
            pass
        print(f'生成的回答是{pred_answer}')
        # dataset='wikitq'
        # args=[]
        
        rouge=load('rouge')

        # score = load(
        #         pred_answer,
        #         gold_answer,)
        rouge_types=['rouge1','rouge2','rougeL','rougeLsum']
        #用rouge_types来规定算哪几个指标不然就默认返回全部四个

        #全部转换为小写
        pred_answer = [p.lower() for p in pred_answer]
        gold_answer = [p.lower() for p in gold_answer]

        try:
            result= rouge.compute(predictions=pred_answer,references=gold_answer, use_aggregator=False,rouge_types=rouge_types)
        except:
            result={'rouge1': [0.0], 'rouge2': [0.0], 'rougeL': [0.0], 'rougeLsum': [0.0]}


        print(f'对于第{idx}个问题，得到的分数是{result}')
        out_put_dict[key] = {
            'gold_answer': gold_answer,
            'pred_answer': pred_answer,
            'score': result,
            'question':ori_question
        }
        
        all.append(result["rouge1"][0])

        end.append(result)
    aver_score=sum(all)/len(all)
    print(f'一共{len(all)}个样本，平均rouge1分数为{aver_score}')

    endtime=time.time()

 
    #保存结果到json文件
    with open(r'D:\\project\\gittest.text\\TbaleReasoningWithLLM\\test\\evaluate\\result\\fetaqa_Qwen3-14B_2025_7_21_22_29.json', 'w', encoding='utf-8') as f:
        json.dump(out_put_dict, f, indent=4, ensure_ascii=False)
    
    print(f'一共耗时 {endtime-start}s  分数--{end}')