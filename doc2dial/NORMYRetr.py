#THIS BACKWARDLY UPDATES THE SCORES OF PASSAGES
import logging, sys
logging.disable(sys.maxsize)
import lucene
import os
from org.apache.lucene.store import MMapDirectory, SimpleFSDirectory, NIOFSDirectory
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.index import FieldInfo, IndexWriter, IndexWriterConfig, IndexOptions, DirectoryReader
from org.apache.lucene.search import IndexSearcher, BoostQuery, Query
from org.apache.lucene.search.similarities import BM25Similarity
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer, util
import json
import pke
from nltk.corpus import stopwords
import re
import spacy

#load dialogue
dial = []
with open('datasets/doc2dial/doc2dial_orqa_format.json') as f:
    dial = json.load(f)

stoplist = stopwords.words('english')
window = 4
use_stems = False
threshold = 0.8
n_yake_words = 5
nlp = spacy.load('en_core_web_trf')
sentence_model = SentenceTransformer('stsb-roberta-large')

LAMBDA = 1

lucene.initVM(vmargs=['-Djava.awt.headless=true'])
searchDir = NIOFSDirectory(Paths.get('doc2dial_lucene_index/'))
searcher = IndexSearcher(DirectoryReader.open(searchDir))
searcher.setSimilarity(BM25Similarity(1.2, 0.75))

outermostlist = [] # main json file jar moddhe shob dhukbe

passencode = {}

def retrieve_passage(qstn, searcher):
    parser = QueryParser('Context', StandardAnalyzer())
    query = parser.parse(qstn)
    topDocs = searcher.search(query, 10).scoreDocs

    topkdocs = []
    for hit in topDocs:
        doc = searcher.doc(hit.doc)
        topkdocs.append({
            "score": hit.score,
            "text": doc.get("Context")
        })
    return topkdocs

def process_yake(passage):
    passage = re.sub(r'[^\w\s]','',passage)
    extractor = pke.unsupervised.YAKE()
    extractor.load_document(input=nlp(passage),
            language='en',
            normalization=None)
    extractor.candidate_selection(n=1)
    extractor.candidate_weighting(window=window,
                use_stems=use_stems)
    
    tmp_keyphrases = extractor.get_n_best(n=n_yake_words, threshold=threshold)
    keyphrases = []
    ordered_keyp = []
    for o in range(0, len(tmp_keyphrases)):
        keyphrases.append(tmp_keyphrases[o][0])
    query_split = passage.split()
    keyp_dicc = {}
    for p in keyphrases:
        keyp_dicc[p] = 1
    for q in query_split:
        if q.lower() in keyp_dicc.keys():
            ordered_keyp.append(q)
    reform_pass = ' '.join(ordered_keyp)

    return reform_pass

def compare_encodings(pass1, pass2):
    if pass1 in passencode:
        pass1_encode = passencode[pass1]
    else:
        pass1_encode = sentence_model.encode(pass1, convert_to_tensor=True)
        passencode[pass1] = pass1_encode

    if pass2 in passencode:
        pass2_encode = passencode[pass2]
    else:
        pass2_encode = sentence_model.encode(pass2, convert_to_tensor=True)
        passencode[pass2] = pass2_encode
    sim = util.pytorch_cos_sim(pass1_encode, pass2_encode)
    return sim.item()
    

yake_map = {}
passmap = {}
#parsing dialogue
for d in range(0, len(dial)): #iterate every dialogue
    print(d)
    innerjson = {}

    query = []
    querystr = ''
    curr_question = dial[d]['question']
    anslist = []
    answer = dial[d]['answer']['text']
    anslist.append(answer)
    replaced_answer = answer.replace(" ", "")
    lower_replaced_answer = replaced_answer.lower()
    if lower_replaced_answer == 'cannotanswer' or lower_replaced_answer == 'notrecovered':
        innerjson['question'] = curr_question
        innerjson['answers'] = anslist
        innerjson['top_ten_passages'] = []
        outermostlist.append(innerjson)
        continue

    history = dial[d]['history'] #its a list
    historytext = []
    allpassages = []
    if history:
        for h in range(0, len(history)):
            historytext.append(history[h]['question'])
    historytext.append(curr_question)

    if historytext:
        for h in range(0, len(historytext)): # appending all the history
            innerretlist = []
            curr_history = historytext[h]
            if h < len(historytext) - 1: # sesher ta baade
                if curr_history in yake_map:
                    reform_curr_history = yake_map[curr_history]
                else:
                    reform_curr_history = process_yake(curr_history)
                    yake_map[curr_history] = reform_curr_history
            else:
                curr_history_2 = re.sub(r'[^\w\s]','',curr_history)
                reform_curr_history = curr_history_2
            
            query.append(reform_curr_history)
            querystr = ' '.join(query)
            if querystr in passmap:
                retrieved = passmap[querystr]
            else:
                qstn2 = querystr
                if not qstn2 or qstn2.isspace():
                    qstn2 = 'null'
                    querystr = 'null'
                retrieved = retrieve_passage(qstn2, searcher) # returns 10 retrieved passages
                passmap[querystr] = retrieved
            for m in range(0, len(retrieved)):
                innerretlist.append((retrieved[m]['text'], retrieved[m]['score'], h, -1)) # innerretlist = [(passage, score, historyturn, simfactor)] 

            if h > 0: #update all previous scores
                for m in range(len(allpassages)-1 , -1, -1):
                    passage = allpassages[m][0]
                    score = allpassages[m][1]
                    score = max(score-LAMBDA, 0)
                    historyturn = allpassages[m][2]
                    if historyturn == h - 1:
                        sumssim = 0
                        for i in range(0, len(innerretlist)):
                            nextpass = innerretlist[i][0]
                            cur_sim = compare_encodings(passage, nextpass)
                            sumssim += cur_sim
                        avg_sim = sumssim / len(innerretlist)
                        score *= avg_sim
                    allpassages[m] = (passage, score, historyturn, avg_sim) # Update the item
            allpassages.extend(innerretlist)
    allpassages = sorted(allpassages, key = lambda x : x[1], reverse = True)

    top_10 = []
    top10cntr = 0
    topmap = {}
    for t in range(0, len(allpassages)):
        if top10cntr == 10:
            break
        if allpassages[t][0] in topmap: # to avoid duplicates
            continue
        topmap[allpassages[t][0]] = True
        top_10.append(allpassages[t][0])
        top10cntr += 1
    
    innerjson['curr_question'] = curr_question
    innerjson['history'] = history
    innerjson['answers'] = anslist
    innerjson['top_ten_passages'] = top_10
    outermostlist.append(innerjson)

output_dir = 'history_models/doc2dial/'
if not os.path.exists(output_dir):
        os.makedirs(output_dir)
with open('history_models/doc2dial/normy_retr.json', 'w') as outfile:
    json.dump(outermostlist, outfile, indent = 4)