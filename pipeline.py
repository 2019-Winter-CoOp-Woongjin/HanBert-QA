# -*- coding: utf-8 -*- 
import sys
import torch
import time
import json
import argparse

import numpy as np
from transformers import BertConfig, BertModel
from trainer import Trainer
from data_loader import SquadExample, SquadResult, squad_convert_examples_to_features
from tokenization_hanbert import HanBertTokenizer
from torch.utils.data import DataLoader
from utils import compute_predictions_logits
from eval_metrics import kor_to_num, get_sent_embedding, cos_sim
from evaluate import f1_score

from flask import Flask, request
from collections import OrderedDict

app = Flask(__name__)

def get_answer(context,question,q_type):
    qas_id = 0

    examples = [
        SquadExample(
        title=None,question_text=question,context_text=context,qas_id=qas_id,
        answer_text=None,start_position_character=None)
    ]
    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=512,
        doc_stride=64,
        max_query_length=64,
        is_training=False,
        threads=1,
    )

    all_results = []
    eval_dataloader = DataLoader(dataset, batch_size=8)
    for batch in eval_dataloader:
        batch = tuple(t.to("cpu") for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            example_indices = batch[3]

            outputs = trainer.model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [output[i].detach().cpu().tolist() for output in outputs]

                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

                all_results.append(result)

    predictions, n_best = compute_predictions_logits(
                        all_examples=examples,
                        all_features=features,
                        all_results=all_results,
                        n_best_size=3,
                        max_answer_length=20,
                        output_nbest_file=None,
                        output_prediction_file=None,
                        tokenizer=tokenizer,
                    )
    

    answer = predictions[qas_id]
    n_best = n_best[qas_id][0]
    logits = n_best['start_logit']+n_best['end_logit']


    index = [n_best['doc_start'],n_best['doc_end']]

    return answer, logits, index



@app.route("/", methods=["POST"])
def index():
    context = request.json["text"]
    question = request.json["question"]
    q_type = int(request.json["q_type"])

    # 객관식
    if q_type == 1:
        answer = request.json["answer"]
        selection = int(answer["selection"])
        choices = [choice for choice in answer["choices"]]

    elif q_type ==2:
        answer = request.json["answer"]
    else:
        answer = request.json["answer"]
    
    process_time = time.time()
    bert_answer, logit, index = get_answer(context, question, q_type)
    print("데이터 처리 시간: {}".format(time.time() - process_time), file=sys.stdout)
    print("logits: {}".format(logit))

    send = OrderedDict()
    send["matched"] = False if logit < 4.0 else True
    send["logit"] = logit

    if q_type == 1:
        bert_choice = 0
        bert_sim = np.array([0,0,0,0],dtype=float)
        # f1_scores = np.array([0,0,0,0],dtype=float)
        
        bert_answer_embed = get_sent_embedding(bert, tokenizer, bert_answer)
        for i, choice in enumerate(choices):
            choice = kor_to_num(choice)
            choice_embed = get_sent_embedding(bert, tokenizer,choice)
            bert_sim[i] = cos_sim(choice_embed, bert_answer_embed)
            # f1_scores[i] = f1_score(choice,bert_answer)
        
        # # 거의 비슷하다면 f1 score로
        # if f1_scores[f1_scores.argmax()] >= 0.85:
        #     print('choose f1')
        #     bert_choice = int(f1_scores.argmax()+1)
        #     send["answer"] = bert_choice
        #     send["prob"] = float(f1_scores[f1_scores.argmax()])
            
        # # 아니면 sentence similarity로    
        # else:
        #     print('choose sent sim')
            bert_choice = int(bert_sim.argmax()+1)
            send["answer"] = bert_choice
            send["prob"] = float(bert_sim[bert_sim.argmax()])
            
            
        send["is_correct"] = True if bert_choice == selection else False


    elif q_type == 2:
        bert_answer = kor_to_num(bert_answer)
        answer = kor_to_num(answer)
        # f1_score_ = f1_score(answer,bert_answer)

        bert_answer_embed = get_sent_embedding(bert, tokenizer, bert_answer)
        answer_embed = get_sent_embedding(bert, tokenizer, answer)
        bert_sim = cos_sim(bert_answer_embed,answer_embed)
        # 거의 비슷하다면 f1 score로
        # if f1_score_ >= 0.85:
        #     print('choose f1')
        #     send["prob"] = f1_score_
        #     send["is_correct"] = True 

        # # 아니면 sentence similarity로    
        # else:
        print('choose sent sim')
        send["prob"] = float(bert_sim)
        send["is_correct"] = True if send["prob"] > 0.8 else False

        send["answer"] = bert_answer
        send["index"] = index
        
        

    else:
        bert_answer = kor_to_num(bert_answer)
        send["answer"] = int(answer == bert_answer)
        send["is_correct"] = False
        send["prob"] = 0.5

    print("answer",answer)
    print(send)
    return json.dumps(send)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model_type
    parser.add_argument(
        "--model_name_or_path", default="HanBert-54kN", type=str,
        help="Path to pre-trained model or shortcut name selected",
    )
    parser.add_argument(
        "--model_dir", default="output/checkpoint-6127", type=str,
        help="Path to load model",
    )
    parser.add_argument(
        "--device", default="output/checkpoint-6127", type=str,
        help="The device which you want to run (gpu/cpu)",
    )
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and args.device=="gpu" else "cpu")

    print("This is very long pre-loading", file=sys.stdout)
    tokenizer = HanBertTokenizer.from_pretrained('HanBert-54kN')
    bert =  BertModel.from_pretrained('HanBert-54kN')
    load_start = time.time()
    trainer = Trainer(args, tokenizer)
    trainer.load_model()  
    trainer.model.eval()
    print("Pre-loading complete. %.2f" % (time.time()-load_start), file=sys.stdout)
    
    app.run(host="127.0.0.1", port="10000")



