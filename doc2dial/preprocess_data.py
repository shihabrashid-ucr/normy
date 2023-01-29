## To convert doc2dial dataset and data into ORQA format
import json
domains = ['dmv', 'ssa', 'va', 'studentaid']

with open('datasets/doc2dial/doc2dial_dial_validation.json') as g:
    dial = json.load(g)
with open('datasets/doc2dial/doc2dial_doc.json') as f:
    document = json.load(f)
orqa = []

i = 0
for l in domains: #every domain
    for k in dial['dial_data'][l].keys(): #every document
        print(i)
        i += 1
        for n in range(0, len(dial['dial_data'][l][k])): #every dialogue
            history = []
            dialid = dial['dial_data'][l][k][n]['dial_id']
            doc_id = dial['dial_data'][l][k][n]['doc_id']
            qid_cntr = 1
            for m in range(0, len(dial['dial_data'][l][k][n]['turns']) - 1): # every turn
                turnjson = {}
                role = dial['dial_data'][l][k][n]['turns'][m]['role']
                if role == 'user':
                    utterance = dial['dial_data'][l][k][n]['turns'][m]['utterance']
                    turnjson['qid'] = dialid + '_' + str(qid_cntr)
                    qid_cntr += 1
                    turnjson['question'] = utterance
                    answer = ''
                    spanlist = []
                    for a in range(0, len(dial['dial_data'][l][k][n]['turns'][m+1]['references'])): #to get all answer spans from dialogue data
                        spanlist.append(dial['dial_data'][l][k][n]['turns'][m+1]['references'][a]['sp_id'])
                    for sp in spanlist: # merging all text spans from document data
                        answer = answer + document['doc_data'][l][doc_id]['spans'][sp]['text_sp'] + ' '
                    ansjson = {}
                    ansjson['text'] = answer
                    turnjson['answer'] = ansjson
                    #print(f'turn: {m+1} len(history) : {len(history)}')
                    turnjson['history'] = history.copy()
                    historyjson = {}
                    historyjson['question'] = utterance
                    historyjson['answer'] = ansjson
                    history.append(historyjson)
                    orqa.append(turnjson)

with open('datasets/doc2dial/doc2dial_orqa_format.json', 'w') as outfile:
    json.dump(orqa, outfile, indent = 4)