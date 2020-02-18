#!/usr/bin/env python
# coding: utf-8

import os
import logging
import timeit
from tqdm import tqdm, trange

import numpy as np
import torch
from torch import nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from tokenization_hanbert import BasicTokenizer
from model import BertForQuestionAnswering
from utils import set_seed, compute_predictions_logits, squad_evaluate
from transformers import BertConfig, BertModel
from data_loader import load_and_cache_examples, SquadV1Processor, SquadResult


logger = logging.getLogger(__name__)

class Trainer(object):
    def __init__(self, args,tokenizer):
        self.args = args
        self.tokenizer = tokenizer

        self.bert_config = BertConfig.from_pretrained(args.model_name_or_path)
        self.model = BertForQuestionAnswering(self.bert_config, args)

        # GPU or CPU
        self.model.to(args.device)

    def train(self):
        args = self.args
        
        train_dataset = load_and_cache_examples(args, self.tokenizer, evaluate=False, output_examples=False)

        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        num_warmup_steps = t_total * args.warmup_proportion
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
        )

        # Check if saved optimizer or scheduler states exist
        if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

        # multi-gpu training
        if args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            args.train_batch_size
            * args.gradient_accumulation_steps
            ,
        )
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 1
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if os.path.exists(args.model_name_or_path):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                logger.info("  Starting fine-tuning.")

        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch")
        
        # Added here for reproductibility
        set_seed(args)

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                self.model.train()
                batch = tuple(t.to(args.device) for t in batch)

                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "start_positions": batch[3],
                    "end_positions": batch[4],
                }


                outputs = self.model(**inputs)
                # model outputs are always tuple in transformers (see doc)
                loss = outputs[0]

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1
                        
                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        self.save_model(global_step)

                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break
            if args.max_steps > 0 and global_step > args.max_steps:
                train_iterator.close()
                break
        
        self.args.model_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
        self.save_model(global_step)
        return global_step, tr_loss / global_step

    def evaluate(self):
        args = self.args
        eval_dataset, examples, features = load_and_cache_examples(args, self.tokenizer, evaluate=True, output_examples=True)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu evaluate
        if args.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)

        # Eval!
        if not os.path.exists(args.model_dir):
            raise Exception("Model doesn't exists! Train first!")
            
        model_name = (args.model_dir).split("/")[-1]
        print("model name: {}".format(model_name))
        
        logger.info("***** Running evaluation {} *****".format(model_name))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_results = []
        start_time = timeit.default_timer()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }
                example_indices = batch[3]
                
                outputs = self.model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [output[i].detach().cpu().tolist() for output in outputs]

                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

                all_results.append(result)

        evalTime = timeit.default_timer() - start_time
        logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(eval_dataset))

        # Compute predictions
        output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(model_name))
        output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(model_name))

        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            self.tokenizer,
            output_prediction_file,
            output_nbest_file,
        )

        # Compute the F1 and exact scores.
        results = squad_evaluate(examples, predictions)
        logger.info(results)
        
        return results
        
    def save_model(self,global_step=None):
        # Save model checkpoint (Overwrite)
        if global_step==None:
            output_dir = os.path.join(self.args.output_dir)
        else:
            output_dir = os.path.join(self.args.output_dir, "checkpoint-{}".format(global_step))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, 'training_config.bin'))
        logger.info("Saving model checkpoint to %s", output_dir)

        
    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.bert_config = BertConfig.from_pretrained(self.args.model_dir)
            logger.info("***** Config loaded *****")
            self.model = BertForQuestionAnswering.from_pretrained(self.args.model_dir, config=self.bert_config, args=self.args)
            self.model.to(self.args.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")
    