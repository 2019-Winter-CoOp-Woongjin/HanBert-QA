import numpy as np
from konlpy.tag import Mecab
from transformers import BertModel, BertConfig
from tokenization_hanbert import HanBertTokenizer
import torch
# use mecab
mecab = Mecab()


units = {'십':10,'백':100,'천':1000,'만':10000,'억':100000000,'조':1000000000000 }
non_units = {'영':0,'일':1,'이':2,'삼':3,'사':4,'오':5,'육':6,'칠':7,'팔':8,'구':9,
    '하나':1,'둘':2,'셋':3,'넷':4,'다섯':5,'여섯':6,'일곱':7,'여덟':8,'아홉':9,
    '십':10,'백':100,'천':1000,'만':10000,'억':100000000,'조':1000000000000
    }
kor_num = {'열':10,'스물':20,'서른':30,'마흔':40,'쉰':50,'예순':60,'일흔':70,'여든':80,'아흔':90}
mm_num = {
    '한':1,
    '두':2,
    '세':3,
    '네':4,
    '스무':20,
}

def kor_to_num(sentence):
    """
    korean number to arabic number
    """
    def other_tag(tag):
        return tag != 'MM' and tag!= 'NR' and tag!='SN'
    
    sent_ = sum([mecab.pos(word)+[(' ','SY',)] for word in sentence.split()],[])
    sent = []
    for word,pos in sent_:
        if pos == 'NR' and (word not in non_units and word not in kor_num) and len(word)>=2:
            word_ = [(word[i],pos) for i in range(len(word))]
            sent.extend(word_)
        else:
            sent.append((word,pos))


    temp_number = 0
    return_sent = ""
    
    try:
        for i,(word,tag) in enumerate(sent):
            if other_tag(tag):
                if temp_number:
                    if i<len(sent)-1 and not other_tag(sent[i+1][1]):
                        continue
                    else:
                        return_sent+= (str(temp_number)+word)
                        temp_number=0
                else:
                    return_sent+=word
                continue

            # 현재 word가 unit이면 (ex. 백/천/만/억 ...)
            if tag=='NR' and word in units:
                if i==0: temp_number = units[word]
                for j in range(i-1,-1,-1):
                    prev_word = sent[j][0]
                    prev_tag = sent[j][1]

                    # 단어가 끝날 때 까지 숫자가 없다면 해당 unit값 그대로
                    if prev_word==' ':
                        temp_number = units[word]
                        break

                    # 이전 단어가 unit이면 비교
                    elif prev_word in units:
                        # 큰 unit이면 곱하기 (ex. 백만)
                        if units[prev_word] < units[word]:
                            temp_number *= units[word]
                        # 작은 unit이면 더하기 (ex. 천백)
                        else:
                            temp_number += units[word]
                        break
                    # 이전 단어가 배수면 곱하기   
                    elif prev_word in non_units:
                        temp_number += non_units[prev_word]*units[word]
                        break
                    elif prev_tag == 'SN':
                        temp_number += int(prev_word)*units[word]
                        break

            elif tag == 'NR' and word in kor_num:
                temp_number += kor_num[word]

            elif (tag=='SN'or tag=='NR') and other_tag(sent[i+1][1]):
                temp_number += int(word) if tag=='SN' else non_units[word]

            elif tag == 'MM':
                temp_number += mm_num[word]
                
    except KeyError:
        return sentence
            
    return return_sent.strip()



def cos_sim(A,B):
    if A.shape != B.shape:
        print("Shape is not same")

    return np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B)) 


def get_sent_embedding(model, tokenizer, sentence):
    sent_input = torch.LongTensor([tokenizer.encode(sentence)])
    sent_mask = torch.LongTensor([[1] * len(sent_input)])
    
    # get embedding
    model.eval()
    with torch.no_grad():
        sent_outputs = model(sent_input,sent_mask,)
    
    seqs = sent_outputs[0][0] # last layer
    
    # layer x batch x tokens x hidden states
    # For each token in the sentence...
    tokens_vec = [token for token in seqs]
    
    return sum(tokens_vec)/len(sent_input)


# Levenshtein distance
def levenshtein_rate(s1,s2, debug=False):
    
    """
    d[i,j] = min(
             d[i-1,j] + deletion cost,
             d[i,j-1] + insertion cost,
             d[i-1,j-1] + substitution cost
            )
    """
    if len(s1) < len(s2):
        return levenshtein(s2,s1, debug)
    
    if len(s2) == 0:
        return len(s1)
    
    prev_row = range(len(s2)+1)
    for i, c1 in enumerate(s1):
        cur_row = [i+1]
        for j, c2 in enumerate(s2):
            insert = prev_row[j+1] + 1
            delete = cur_row[j] + 1
            substitute = prev_row[j] + (c1!=c2)
            cur_row.append(min(insert,delete,substitute))
            
        if debug:
            print(cur_row[1:])
        
        prev_row = cur_row
    
    return prev_row[-1]/len(s1)
 
# 명사 자카드 지수
def nouns_prob(answer, prediction):
    nanswer = set(mecab.nouns(answer))
    npred = set(mecab.nouns(prediction))
    intersection = nanswer & npred
    union = nanswer | npred

    if len(union)==0:
        return -1
    return len(intersection) /  len(union)
    
