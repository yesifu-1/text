import json
def extract() -> list:
    
    dataset=[]

    with open("simple_Table_reasoning_test/input.json","r") as f:
        
        lines = json.load(f)["data"]
        
        for i,line in enumerate(lines):
            if i<15:
                dic = line
                feta_id = dic['feta_id']
                caption = dic['table_page_title']
                question = dic['question']
                answer = dic["answer"]
                sub_title = dic['table_section_title']
                header = dic['table_array'][0]
                rows = dic['table_array'][1:]
                data = {
                "id": feta_id,
                "table": {
                    "id": feta_id,
                    "header": header,
                    "rows": rows,
                    "page_title": caption,
                },
                "question": question,
                "answer": answer
                }
                dataset.append(data)
            else:
                pass
    return dataset

