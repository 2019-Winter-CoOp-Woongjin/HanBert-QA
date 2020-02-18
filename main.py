import torch
import argparse
import logging
import os
import glob
import random
import timeit

import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from trainer import Trainer
from utils import set_seed, init_logger
from tokenization_hanbert import HanBertTokenizer


logger = logging.getLogger(__name__)
def main(args):
    init_logger()
    tokenizer = HanBertTokenizer.from_pretrained(args.model_name_or_path)
        
    trainer = Trainer(args, tokenizer)
    
    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        global_step, tr_loss = trainer.train()
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    
    if args.do_eval:
        trainer.load_model()
        trainer.evaluate()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model_type
    parser.add_argument(
        "--model_name_or_path", default="HanBert-54kN", type=str,
        help="Path to pre-trained model or shortcut name selected",
    )
    parser.add_argument(
        "--output_dir", default="./output", type=str,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--data_dir", default="./data", type=str,
        help="The input data dir. Should contain the .json files for the task."
    )
    parser.add_argument("--model_dir", default=None, type=str, help="Path to save, load model")
    parser.add_argument(
        "--train_filename", default="KorQuAD_v1.0_train.json", type=str, help="The input training file. If a data dir is specified, will look for the file there"
    )
    parser.add_argument(
        "--predict_filename", default="KorQuAD_v1.0_dev.json", type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
    )
    parser.add_argument(
        "--cache_dir", default="./cache", type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--null_score_diff_threshold", type=float, default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )
    parser.add_argument(
        "--max_seq_length", default=512, type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride", default=64, type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64, type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=float, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_steps", default=-1, type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_proportion", default=0.0, type=float, help="Linear warmup proportion")
    parser.add_argument(
        "--n_best_size", default=5, type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length", default=10, type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=2000, help="Save checkpoint every X updates steps.")
    
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
    args = parser.parse_args()

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )
    args = parser.parse_args()
        
    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
        
    args.device = device
        
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning(
        "Device: %s, n_gpu: %s",
        device,
        args.n_gpu,
    )
    
    # Set seed
    set_seed(args)
        
    main(args)