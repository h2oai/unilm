# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003 (Bert or Roberta). """
from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import math
import pandas as pd

import numpy as np
import torch
# from seqeval.metrics import (
#     precision_score,
#     recall_score,
#     f1_score,
#     classification_report,
# )
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from layoutlm.utils_seq_labeling import (
    convert_examples_to_features,
    get_labels,
    read_examples_from_file,
)
from layoutlm.modeling_layoutlm import LayoutLMForTokenClassification
from transformers import (
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
)
from transformers import RobertaConfig, RobertaForTokenClassification, RobertaTokenizer
from transformers import (
    DistilBertConfig,
    DistilBertForTokenClassification,
    DistilBertTokenizer,
)

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig, RobertaConfig, DistilBertConfig)
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
    "distilbert": (
        DistilBertConfig,
        DistilBertForTokenClassification,
        DistilBertTokenizer,
    ),
    "layoutlm": (BertConfig, LayoutLMForTokenClassification, BertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix=""):
    eval_dataset = load_and_cache_examples(
        args, tokenizer, labels, pad_token_label_id, mode=mode
    )

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = (
        SequentialSampler(eval_dataset)
        if args.local_rank == -1
        else DistributedSampler(eval_dataset)
    )
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    )

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    raw_preds = None
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Entity recognition"):
        if args.model_type != "layoutlm":
            batch = batch[:4]
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
            }
            if args.model_type == "layoutlm":
                inputs["bbox"] = batch[4]
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "layoutlm"] else None
                )  # XLM and RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            if args.n_gpu > 1:
                tmp_eval_loss = (
                    tmp_eval_loss.mean()
                )  # mean() to average on multi-gpu parallel evaluating

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            raw_preds = preds
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            raw_preds = preds
            out_label_ids = np.append(
                out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
            )

    # eval_loss = eval_loss / nb_eval_steps
    sigmoid_v = np.vectorize(sigmoid)
    sigmoid_preds = sigmoid_v(raw_preds)
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = []
    probabilities = []

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list.append(label_map[preds[i][j]])
                probabilities.append(sigmoid_preds[i][j])

    probs = pd.DataFrame(probabilities)

    probs.columns = labels
    probs['labels'] = preds_list

    return probs


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Only first process in distributed training process the dataset. Others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}".format(
            mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = read_examples_from_file(args.data_dir, mode)
        features = convert_examples_to_features(
            examples,
            labels,
            args.max_seq_length,
            tokenizer,
            cls_token_at_end=bool(args.model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences,
            # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            pad_token_label_id=pad_token_label_id,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Only the first process in distributed training process the dataset. Others will use the cache
        torch.distributed.barrier()

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_bboxes = torch.tensor([f.boxes for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_bboxes,
    )
    return dataset


def arguments():
    parser = argparse.ArgumentParser(description='layoutlm entity recognition')

    ####################################################################################################################
    # args
    ####################################################################################################################

    # Required parameters
    parser.add_argument("--data_dir", default='test_data', type=str,
                        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.")
    parser.add_argument("--model_type", default='layoutlm', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default='layoutlm_base', type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--output_dir", default='output_base', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Other parameters
    parser.add_argument(
        "--labels",
        default="data/labels.txt",
        type=str,
        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",
    )
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--do_train",
        default=False,
        # action="store_true",
        help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval",
        default=False,
        # action="store_true",
        help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_predict",
        default=True,
        # action="store_true",
        help="Whether to run predictions on the test set.",
    )
    parser.add_argument(
        "--evaluate_during_training",
        # default=True,
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        default=True,
        # action="store_true",
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=50.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )

    parser.add_argument(
        "--logging_steps", type=int, default=50, help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=-1,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--no_cuda",
        default=True,
        # action="store_true",
        help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        default=True,
        # action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        default=True,
        # action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--server_ip", type=str, default="", help="For distant debugging."
    )
    parser.add_argument(
        "--server_port", type=str, default="", help="For distant debugging."
    )

    args = parser.parse_args(args=[])

    return args


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def predict_entity(data_dir='layoutlm/test_data',
                   model_type='layoutlm',
                   model_name_or_path='weights/layoutlm_base',
                   do_lower_case=True,
                   max_seq_length=512,
                   do_predict=True,
                   output_dir='layoutlm_output',
                   labels='layoutlm/data/labels.txt',
                   no_cuda=True):
    args = arguments()

    args.data_dir = data_dir
    args.model_type = model_type
    args.model_name_or_path = model_name_or_path
    args.do_lower_case = do_lower_case
    args.max_seq_length = max_seq_length
    args.do_predict = do_predict
    args.output_dir = output_dir
    args.labels = labels
    args.no_cuda = no_cuda

    # ===================================================================================================================
    # Setup CUDA, GPU & distributed training
    # ==================================================================================================================
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        # torch.cuda.set_device(device)
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    logger.info(f'layoutlm is using: {device}')

    # ==================================================================================================================

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    labels = get_labels(args.labels)
    pad_token_label_id = CrossEntropyLoss().ignore_index
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    ####################################################################################################################
    # Predict on test
    ####################################################################################################################

    predictions = None
    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path, do_lower_case=args.do_lower_case
        )
        model = model_class.from_pretrained(args.model_name_or_path)
        model.to(args.device)
        predictions = evaluate(
            args, model, tokenizer, labels, pad_token_label_id, mode="test"
        )

    return predictions


if __name__ == "__main__":
    predict_entity(data_dir='test_data',
                   model_type='layoutlm',
                   model_name_or_path='../weights/layoutlm_base',
                   do_lower_case=True,
                   max_seq_length=512,
                   do_predict=True,
                   output_dir='layoutlm_output',
                   labels='data/labels.txt',
                   no_cuda=True)
