import json
import pickle

with open('datasets/convinse/train_set_ALL.json') as f:
    data = json.load(f)

with open('datasets/convinse/wikipedia_dump.pickle', 'rb') as infile:
    corpus_dicc = pickle.load(infile)

outer = []
cntr = 0
for d in data:
    print(cntr)

    his = []
    for q in d['questions']:
        qid = q['question_id']
        qtext = q['question']
        ans = q['answer_text']
        ans_id = q['answers'][0]['id']
        src = q['answer_src']
        entity_id = q['entities'][0]['id']
        
        pos_cntxt = []
        if src == 'text':
            if entity_id in corpus_dicc:
                for j in corpus_dicc[entity_id]:
                    check = False
                    if j['source'] == 'text':
                        for k in j['wikidata_entities']:
                            if k['id'] == ans_id:
                                pos_cntxt = [{'text' : j['evidence_text']}]
                                check = True
                                break
                        if check:
                            break
        else:
            pos_cntxt = []

        inner = {}
        inner['qid'] = qid
        inner['question'] = qtext
        inner['answer'] = {'text' : ans}
        inner['history'] = his.copy()
        inner['positive_ctxt'] = pos_cntxt
        inner['ans_source'] = src
        outer.append(inner)

        hisjson = {}
        hisjson['question'] = qtext
        hisjson['answer'] = {'text' : ans}
        his.append(hisjson)
    cntr += 1

    with open('datasets/convinse/convinse_orqa_format.json', 'w') as outfile:
        json.dump(outer, outfile, indent = 4)