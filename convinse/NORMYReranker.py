import json
from collections import deque
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

with open('history_models/convinse/normy_retr.json') as f:
    dial = json.load(f)

tokenizer = AutoTokenizer.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")
model = AutoModelForSequenceClassification.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")

outermostlist = []
HISTORY_TOKEN_THRESHOLD = 100
for d in range(0, len(dial)):
    print(d)
    innerjson = {}
    query = deque()
    querystr = ''
    anslist = []
    answer = dial[d]['answer']
    anslist.append(answer)
    retrieved = dial[d]['top_ten_passages']

    if not retrieved:
        innerjson['curr_question'] = dial[d]['question']
        innerjson['answer'] = anslist
        innerjson['history'] = history
        innerjson['reranked_retr'] = []
        outermostlist.append(innerjson)
        continue
    curr_question = dial[d]['curr_question']
    history = dial[d]['history'] #its a list
    retr_score_mapping = {}
    tokensize = 0
    tokenmap = {}
    if history:
        query.append(history[0]['question'])
        for h in range(1, len(history)): # appending all the history
            if len(history) <= 6: # we will consider window = 6, change w here
                query.append(history[h]['question'])
            else:
                if h < len(history)-6:
                    continue
                else:
                    query.append(history[h]['question'])
        query.append(curr_question)
        querystr = ' '.join(query)
    else:
        querystr = curr_question
    
    querystrlist = [querystr] * len(retrieved)
    retrievedlist = []
    for r in retrieved:
        retrievedlist.append(r[0]) # cause in retrieved there was score also
        retr_score_mapping[r[0]] = r[1]
    features = tokenizer(querystrlist, retrievedlist,  padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        scores = model(**features).logits
        scores = scores.detach().cpu().numpy()
    rerankedlist = []
    for r in range(0, len(retrieved)):
        #print(f'scores: {scores[r][1]}')
        rerankedlist.append((retrieved[r][0], float(retr_score_mapping[retrieved[r][0]]) ,float(scores[r][1]))) # [passage, retr_score, rerank_score]
    rerankedlist = sorted(rerankedlist, key = lambda item : item[2], reverse=True)
    #print(rerankedlist)

    innerjson['curr_question'] = curr_question
    innerjson['answers'] = anslist
    innerjson['history'] = history
    innerjson['reranked_retr'] = rerankedlist
    outermostlist.append(innerjson)
with open('history_models/convinse/normy_reranker.json', 'w') as outfile:
    json.dump(outermostlist, outfile, indent = 4)
    


