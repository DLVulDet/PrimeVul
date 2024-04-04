from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from tqdm import tqdm, trange
import multiprocessing
from model import ModelWithCLR, ModelWithCLRForClassification
cpu_cont = multiprocessing.cpu_count()
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)
}


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 idx,
                 label,

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx=str(idx)
        self.label=label

        
def convert_examples_to_features(js,tokenizer,args):
    #source
    code=js['func']
    code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens,source_ids,js['idx'],js['target'])

class CLRDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        # we have paired examples (vul, benign) and benign examples
        self.paired_examples = []
        self.benign_examples = []
        self.args = args
        data = []
        with open(file_path) as f:
            for line in f:
                line=line.strip()
                js=json.loads(line)
                data.append(js)
        logger.info("Loading %d examples from %s", len(data), file_path)
        for js in tqdm(data):
            if js["pos_sample"] == "":
                self.benign_examples.append(convert_examples_to_features(js["neg_sample"],tokenizer,args))
            else:
                self.paired_examples.append([convert_examples_to_features(js["pos_sample"],tokenizer,args), convert_examples_to_features(js["neg_sample"],tokenizer,args)])
        if 'train' in file_path:
            for example in self.paired_examples[:3]:
                logger.info("*** Paired Example ***")
                for e in example:
                    logger.info("idx: {}".format(e.idx))
                    logger.info("label: {}".format(e.label))
                    logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in e.input_tokens]))
                    logger.info("input_ids: {}".format(' '.join(map(str, e.input_ids))))
            for example in self.benign_examples[:3]:
                logger.info("*** Benign Example ***")
                logger.info("idx: {}".format(example.idx))
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
        
    def __len__(self):
        return len(self.paired_examples)

    def __getitem__(self, i):
        example = self.paired_examples[i]
        vul_example = example[0]
        benign_example = example[1]
        assert vul_example.label == 1 and benign_example.label == 0
        # sample args.group_size - 2 examples from benign examples
        add_benign_examples = random.sample(self.benign_examples, self.args.group_size - 2)
        add_benign_examples = [e.input_ids for e in add_benign_examples]
        add_benign_labels = [0] * len(add_benign_examples)
        final_input_ids = [vul_example.input_ids, benign_example.input_ids] + add_benign_examples
        final_labels = [1, 0] + add_benign_labels
        # make the final_input_ids and final_labels to be tensors
        final_input_ids = torch.tensor(final_input_ids)
        final_labels = torch.tensor(final_labels)
        assert final_input_ids.shape[0] == final_labels.shape[0] == self.args.group_size
        return final_input_ids, final_labels

class CLSDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        with open(file_path) as f:
            for line in f:
                js=json.loads(line.strip())
                self.examples.append(convert_examples_to_features(js,tokenizer,args))
        if 'train' in file_path:
            # from collections import Counter
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
                    

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def calculate_metrics(labels, preds):
    acc=accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    TN, FP, FN, TP = confusion_matrix(labels, preds).ravel()
    tnr = TN/(TN+FP)
    fpr = FP/(FP+TN)
    fnr = FN/(TP+FN)
    return round(acc,4)*100, round(prec,4)*100, \
        round(recall,4)*100, round(f1,4)*100, round(tnr,4)*100, \
            round(fpr,4)*100, round(fnr,4)*100

def train(args, train_dataset, model, tokenizer):
    """ Train the model """ 
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=args.train_batch_size ,num_workers=8,pin_memory=True)
    args.max_steps=args.epoch*len(train_dataloader)
    args.save_steps=len( train_dataloader)
    args.num_train_epochs=args.epoch
    model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.warmup_steps == 0:
        num_warmup = args.max_steps*args.warmup_ratio
    else:
        num_warmup = args.warmup_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup,
                                                num_training_steps=args.max_steps)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step = args.start_step
    tr_loss, logging_loss,avg_loss,tr_nb,tr_num,train_loss = 0.0, 0.0,0.0,0,0,0
    best_mrr=0.0
    best_f1=0.0
    best_acc=0.0
    patience = 0

    model.zero_grad()
 
    step = 0
    for idx in range(args.start_epoch, int(args.num_train_epochs)): 
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num=0
        train_loss=0
        logits_lst = []
        labels_lst = []
        for local_step, batch in enumerate(bar):
            group_inputs = batch[0].to(args.device)    
            group_labels = batch[1].to(args.device)
            model.train()
            loss, logits, labels = model(group_inputs, group_labels)
            
            logits_lst.append(logits.detach().cpu().numpy())
            labels_lst.append(labels.detach().cpu().numpy())

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num+=1
            train_loss+=loss.item()
            if avg_loss==0:
                avg_loss=tr_loss
            avg_loss=round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))
            
            ###

            if (local_step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb=global_step
            
            ###
            # log after every logging_steps (e.g., 1000)
            ###
            if (step + 1) % args.logging_steps == 0:
                avg_loss=round(train_loss/tr_num,5)
                step_logits_lst=np.concatenate(logits_lst,0)
                step_labels_lst=np.concatenate(labels_lst,0)
                # if isinstance(model, DefectModel) or isinstance(model, GPT2Classifier):
                step_preds_lst=step_logits_lst[:,0]>0.5
                train_acc, train_prec, train_recall, train_f1, train_tnr, train_fpr, train_fnr = calculate_metrics(step_labels_lst, step_preds_lst)
                if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                    results = evaluate(args, model, tokenizer,eval_when_training=True)
                    for key, value in results.items():
                        logger.info("  %s = %s", key, round(value,4))                    
                
                # Save model checkpoint    
                if results['eval_f1']>best_f1:
                    best_f1=results['eval_f1']
                    logger.info("  "+"*"*20)  
                    logger.info("  Best f1:%s",round(best_f1,4))
                    logger.info("  "+"*"*20)                          
                    
                    checkpoint_prefix = f'checkpoint-best-f1/{args.project}/{args.model_dir}'
                    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)                        
                    model_to_save = model.module if hasattr(model,'module') else model
                    output_dir = os.path.join(output_dir, f'model.bin') 
                    torch.save(model_to_save.state_dict(), output_dir)
                    logger.info(f"Saving best f1 model checkpoint at epoch {idx} step {step} to {output_dir}")
            
            # increment step within the same epoch
            step += 1

        ###
        # log after every epoch
        ###
        avg_loss=round(train_loss/tr_num,5)
        logits_lst=np.concatenate(logits_lst,0)
        labels_lst=np.concatenate(labels_lst,0)
        
        if args.local_rank in [-1, 0]:
            if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                results = evaluate(args, model, tokenizer,eval_when_training=True)
                for key, value in results.items():
                    logger.info("  %s = %s", key, round(value,4))                    
            
            # save model checkpoint at ep10
            if idx == 9:
                checkpoint_prefix = f'checkpoint-acsac/{args.project}/{args.model_dir}'
                output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)                        
                model_to_save = model.module if hasattr(model,'module') else model
                output_dir = os.path.join(output_dir, f'model-ep{idx}.bin') 
                torch.save(model_to_save.state_dict(), output_dir)
                logger.info(f"ACSAC: Saving model checkpoint at epoch {idx} to {output_dir}")
            
            # Save model checkpoint    
            if results['eval_f1']>best_f1:
                best_f1=results['eval_f1']
                logger.info("  "+"*"*20)  
                logger.info("  Best f1:%s",round(best_f1,4))
                logger.info("  "+"*"*20)                          
                
                checkpoint_prefix = f'checkpoint-best-f1/{args.project}/{args.model_dir}'
                output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)                        
                model_to_save = model.module if hasattr(model,'module') else model
                output_dir = os.path.join(output_dir, f'model.bin') 
                torch.save(model_to_save.state_dict(), output_dir)
                logger.info(f"Saving best f1 model checkpoint at epoch {idx} to {output_dir}")
                patience = 0
            else:
                patience += 1

            if results['eval_acc']>best_acc:
                best_acc=results['eval_acc']
                logger.info("  "+"*"*20)
                logger.info("  Best acc:%s",round(best_acc,4))
                logger.info("  "+"*"*20)                          
                
                checkpoint_prefix = f'checkpoint-best-acc/{args.project}/{args.model_dir}'
                output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)                        
                model_to_save = model.module if hasattr(model,'module') else model
                output_dir = os.path.join(output_dir, f'model.bin') 
                torch.save(model_to_save.state_dict(), output_dir)
                logger.info(f"Saving best acc model checkpoint at epoch {idx} to {output_dir}")
                patience = 0
            else:
                patience += 1
 

eval_dataset=None
def evaluate(args, model, tokenizer,eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = CLRDataset(tokenizer, args,args.eval_data_file)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[] 
    labels=[]
    for batch in eval_dataloader:
        group_inputs = batch[0].to(args.device)        
        group_labels=batch[1].to(args.device) 
        with torch.no_grad():
            lm_loss,logit,pred_labels = model(group_inputs,group_labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(pred_labels.cpu().numpy())
        nb_eval_steps += 1
    logits=np.concatenate(logits,0)
    labels=np.concatenate(labels,0)
    # if isinstance(model, DefectModel) or isinstance(model, GPT2Classifier):
    preds=logits[:,0]>0.5
    eval_acc, eval_prec, eval_recall, eval_f1, eval_tnr, eval_fpr, eval_fnr = calculate_metrics(labels, preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    result = {
        "eval_loss": float(perplexity),
        "eval_acc": eval_acc,
        "eval_prec": eval_prec,
        "eval_recall": eval_recall,
        "eval_f1": eval_f1,
        "eval_tnr": eval_tnr,
        "eval_fpr": eval_fpr,
        "eval_fnr": eval_fnr,
    }
    return result

def test(args, model, tokenizer):
    eval_dataset = CLSDataset(tokenizer, args,args.test_data_file)


    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]   
    labels=[]
    for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
        inputs = batch[0].to(args.device)        
        label=batch[1].to(args.device) 
        with torch.no_grad():
            logit = model(inputs)
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())

    logits=np.concatenate(logits,0)
    labels=np.concatenate(labels,0)

    preds=logits[:,0]>0.5
    vuln_scores = logits[:,0].tolist()
    os.makedirs(os.path.join(args.output_dir, args.project), exist_ok=True)
    # for the convenience of saving different testing results
    test_project = args.test_project if args.test_project else args.project
    with open(os.path.join(args.output_dir, test_project, "predictions.txt"),'w') as f:
        for example,pred,vs in zip(eval_dataset.examples,preds,vuln_scores):
            if pred:
                f.write(example.idx+f'\t1\t{vs}\n')
            else:
                f.write(example.idx+f'\t0\t{vs}\n')
    # remove redundant preds and labels caused by the batch
    preds = preds[:len(eval_dataset.examples)]
    preds = [1 if p else 0 for p in preds]
    labels = labels[:len(eval_dataset.examples)]
    logger.info(f"# of preds: {len(preds)}, # of labels: {len(labels)}")
    test_acc, test_prec, test_recall, test_f1, test_tnr, test_fpr, test_fnr = calculate_metrics(labels, preds)

    result = {
        "test_acc": test_acc,
        "test_prec": test_prec,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_tnr": test_tnr,
        "test_fpr": test_fpr,
        "test_fnr": test_fnr,
    }
    return result 
                        
                        
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--project', type=str, required=True, help="using dataset from this project.")
    parser.add_argument('--test_project', type=str, required=False, help="test setup name.")
    parser.add_argument('--model_dir', type=str, required=True, help="directory to store the model weights.")

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
                    
    parser.add_argument("--model_type", default="roberta", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="microsoft/unixcoder-base", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--config_name", default="microsoft/unixcoder-base", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="microsoft/unixcoder-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")    
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    # clr specific parameters
    parser.add_argument("--group_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--clr_mask", action='store_true',
                        help="Whether to use CLR mask for training.")
    parser.add_argument("--clr_temp", default=0.05, type=float,
                        help="CLR temperature.")
    
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    

    args = parser.parse_args()


    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.per_gpu_train_batch_size=args.train_batch_size//args.n_gpu
    args.per_gpu_eval_batch_size=args.eval_batch_size//args.n_gpu
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)


    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels=1
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)    
    else:
        model = model_class(config)

    model=ModelWithCLR(model,config,tokenizer,args,clr_mask=args.clr_mask,clr_temp=args.clr_temp)
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = CLRDataset(tokenizer, args,args.train_data_file)
        if args.local_rank == 0:
            torch.distributed.barrier()

        train(args, train_dataset, model, tokenizer)



    # Evaluation
    results = {}
            
    if args.do_test and args.local_rank in [-1, 0]:
        checkpoint_prefix = f'checkpoint-best-f1/{args.project}/{args.model_dir}/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        cls_model = ModelWithCLRForClassification(model.encoder,config,tokenizer,args)
        cls_model.load_state_dict(torch.load(output_dir),strict=False)
        cls_model.to(args.device)
        result = test(args, cls_model, tokenizer)
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))

    return results


if __name__ == "__main__":
    main()


