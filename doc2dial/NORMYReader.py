import json, string, re
from collections import deque, Counter, defaultdict, OrderedDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering
import torch
import numpy as np
import neuralcoref
import spacy
#import logging, sys
#logging.disable(sys.maxsize)

pretrained_model_name_or_path='bert-large-uncased-whole-word-masking-finetuned-squad'
READER_PATH = pretrained_model_name_or_path
tokenizer = AutoTokenizer.from_pretrained(READER_PATH)
model = AutoModelForQuestionAnswering.from_pretrained(READER_PATH)
max_len = model.config.max_position_embeddings
ANS_SIZE_THRESHOLD = 512

def f1_score(prediction, ground_truth):
  prediction_tokens = normalize_answer(prediction).split()
  ground_truth_tokens = normalize_answer(ground_truth).split()
  common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
  num_same = sum(common.values())
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    return re.sub(r'\b(a|an|the)\b', ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def tokenize(question, text, chunked, max_len):
    inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    if len(input_ids) > max_len:
        inputs = chunkify(inputs)
        chunked = True
    return inputs, chunked

def chunkify(inputs):
    """ 
    Break up a long article into chunks that fit within the max token
    requirement for that Transformer model. 

    Calls to BERT / RoBERTa / ALBERT require the following format:
    [CLS] question tokens [SEP] context tokens [SEP].
    """
    # create question mask based on token_type_ids
    # value is 0 for question tokens, 1 for context tokens
    qmask = inputs['token_type_ids'].lt(1) #lt = less than 1, because question are represented by 0s, qmask = questions =[0,0,]
    qt = torch.masked_select(inputs['input_ids'], qmask)
    chunk_size = max_len - qt.size()[0] - 1 # the "-1" accounts for
    # having to add an ending [SEP] token to the end

    # create a dict of dicts; each sub-dict mimics the structure of pre-chunked model input
    chunked_input = OrderedDict()
    for k,v in inputs.items():
        q = torch.masked_select(v, qmask) # q means the question
        c = torch.masked_select(v, ~qmask) #c means the context
        chunks = torch.split(c, chunk_size)
        
        for i, chunk in enumerate(chunks):
            if i not in chunked_input:
                chunked_input[i] = {}

            thing = torch.cat((q, chunk))
            if i != len(chunks)-1:
                if k == 'input_ids':
                    thing = torch.cat((thing, torch.tensor([102])))
                else:
                    thing = torch.cat((thing, torch.tensor([1])))

            chunked_input[i][k] = torch.unsqueeze(thing, dim=0)
    return chunked_input

def get_score(start_scores, end_scores):
    scores = []
    s_soft = torch.nn.Softmax(dim=1)
    e_soft = torch.nn.Softmax(dim=1)
    soft_s = s_soft(start_scores)
    soft_e = e_soft(end_scores)
    s_scores = soft_s.detach().numpy().flatten()
    e_scores = soft_e.detach().numpy().flatten()
    highest_s_scores = np.amax(s_scores)
    highest_e_scores = np.amax(e_scores)
    score = (highest_s_scores + highest_e_scores) / 2
    return score

def get_answer(inputs, chunked, model):
    anslist = []
    if chunked:
        answer = ''
        tokenlist = []
        startscorelist = []
        endscorelist = []
        for k, chunk in inputs.items():
            outputs = model(**chunk)

            answer_start_scores = outputs.start_logits
            answer_end_scores = outputs.end_logits
            answer_start = torch.argmax(answer_start_scores)
            answer_end = torch.argmax(answer_end_scores) + 1
            ans = convert_ids_to_string(chunk['input_ids'][0][answer_start:answer_end])
            tokens = tokenizer.convert_ids_to_tokens(chunk['input_ids'][0])
            tokenlist.append(tokens)
            startscorelist.append(answer_start_scores)
            endscorelist.append(answer_end_scores)
            if not ans:
                ans = '' #if answer is [CLS] that means no answer
            anslist.append(ans)
        return anslist, startscorelist, endscorelist
    else:
        tokenlist = []
        startscorelist = []
        endscorelist = []
        outputs = model(**inputs)

        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits
        answer_start = torch.argmax(answer_start_scores)  # get the most likely beginning of answer with the argmax of the score
        answer_end = torch.argmax(answer_end_scores) + 1  # get the most likely end of answer with the argmax of the score
        
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        tokenlist.append(tokens)
        startscorelist.append(answer_start_scores)
        endscorelist.append(answer_end_scores)
        anslist.append(convert_ids_to_string(inputs['input_ids'][0][
                                            answer_start:answer_end]))
        return anslist, startscorelist, endscorelist

def convert_ids_to_string(input_ids):
    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids))
    
def process_answer(question, paragraph, model, chunked, max_len):
    answerlist = []
    startscorelist = []
    endscorelist = []
    inputs, chunked = tokenize(question, paragraph, chunked, max_len)
    answerlist, startscorelist, endscorelist = get_answer(inputs, chunked, model)
    if len(answerlist) == 1:
        score = get_score(startscorelist[0], endscorelist[0])
        return answerlist[0], score
    else:
        max_score = -1
        max_index = -1
        for i in range(0, len(answerlist)):
            score = get_score(startscorelist[i], endscorelist[i])
            if score > max_score:
                max_score = score
                max_index = i
        return answerlist[max_index], max_score

with open('history_models/doc2dial/normy_reranker.json') as f:
    dial = json.load(f)

nlp = spacy.load('en')
neuralcoref.add_to_pipe(nlp, greedyness=0.5)

outermostlist = []
f1 = 0
for d in range(0, len(dial)):
    print(d)
    innerjson = {}
    query = []
    querystr = ''
    curr_question = dial[d]['curr_question']
    ground_answer = dial[d]['answers'][0]
    history = dial[d]['history']
    passages = dial[d]['reranked_retr']
    historytext = []
    allhistorystr = ''
    if not passages:
        continue
    if history:
        for h in range(0, len(history)): # appending all the history
            curr_history = history[h]['question']
            allhistorystr += ' ' + curr_history
        doc = nlp(allhistorystr + ' ** ' + curr_question)
        #print(allhistorystr + ' ** ' + curr_question)
        resolved = doc._.coref_resolved
        #print(resolved)
        res_split = resolved.split('**')
        querystr = str(res_split[-1])        
    else:
        querystr = curr_question
    possible_answers = []

    for p in passages:
        chunked = False
        pred_ans, score = process_answer(querystr, p[0], model, chunked, max_len)
        possible_answers.append((pred_ans, float(p[1]) + float(p[2]) + float(score)))
    possible_answers = sorted(possible_answers, key = lambda item : item[1], reverse=True)
    f1 += f1_score(possible_answers[0][0], ground_answer)

    innerjson['question'] = querystr
    innerjson['history'] = history
    innerjson['ground_answer'] = possible_answers[0][0]
    innerjson['total_score'] = possible_answers[0][1]
    outermostlist.append(innerjson)
avg_f1 = f1 / len(dial)

print(f'avg f1: {avg_f1}')
