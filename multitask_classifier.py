import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data

from evaluation import model_eval_sst, model_eval_multitask, model_eval_train_multitask, test_model_multitask

from ray import tune
import ray

import pickle
from copy import deepcopy

#ABSOLUTE_PATH_PREFIX = "/Users/ad2we/Desktop/2022_2023_Stanford/winqtr_2023/cs224n_dfp/minbert-default-final-project/"
ABSOLUTE_PATH_PREFIX = "/Users/callumburgess/Documents/classes/CS 224N/minbert-default-final-project/"


NUM_BATCHES_PER_EPOCH = 1000
TQDM_DISABLE = True

# CURRENTLY NOT IN USE
LIMIT_PARAPHRASE_TRAIN_BATCHES = True
PARAPHRASE_MAX_BATCHES = 100

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:
    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config, connected_info, subsets=False):        
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
         
        # baseline - specific dropout + linear layers for each task-specific head:  
        self.stm_dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.stm_linear  = torch.nn.Linear(config.hidden_size, len(config.num_labels))

        self.ppr_dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.ppr_linear  = torch.nn.Linear(config.hidden_size, 1)
        
        self.sim_dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.sim_linear  = torch.nn.Linear(config.hidden_size, config.hidden_size)

        # parameter sharing regime-specific architecture:
        self.connected_info = connected_info

        num_shared          = sum([1 if x=='S' else 0 for x in connected_info])
        num_individual      = sum([3 if x=='I' else 0 for x in connected_info])
        if subsets:
            num_shared          = sum([1 if x=='S' else 0 for x in connected_info])
            num_individual      = sum([3 if x=='I' else 0 for x in connected_info]) + num_shared
            print("BIG MONEY",num_shared, num_individual)
        self.shared         = nn.ModuleList([torch.nn.Linear(config.hidden_size, config.hidden_size) for i in range(num_shared)])
        self.individual     = nn.ModuleList([torch.nn.Linear(config.hidden_size, config.hidden_size) for i in range(num_individual)])
        self.subsets = subsets
        
    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this (e.g., by adding other layers).
        
        # currently returns embedding straight from BERT:
        return self.bert(input_ids, attention_mask)['pooler_output']

    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        pooled = self.bert(input_ids, attention_mask)['pooler_output']
        
        # apply the parameter sharing regime: 
        shared_index = 0
        individual_index = 0
        subset_index = 0 
        for val in self.connected_info:
            if not self.subsets: 
                if val == "S":
                    pooled = self.shared[shared_index](pooled)
                    pooled = nn.ReLU()(pooled)
                    shared_index += 1
                if val == "I":
                    pooled = self.individual[individual_index](pooled)
                    pooled = nn.ReLU()(pooled)
                    individual_index += 3
            else: 
                print(individual_index)
                if val == "S" and self.connected_info[0] != 0:
                    pooled = self.shared[shared_index](pooled)
                    pooled = nn.ReLU()(pooled)
                    shared_index += 1
                if val == "S" and self.connected_info[0] == 0:
                    pooled = self.individual[-1-subset_index](pooled)
                    pooled = nn.ReLU()(pooled)
                    subset_index += 1
                if val == "I":
                    pooled = self.individual[individual_index](pooled)
                    pooled = nn.ReLU()(pooled)
                    individual_index += 3

        logits = self.stm_linear(self.stm_dropout(pooled))
        return F.log_softmax(logits, dim=1)


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        pooled1, pooled2 = self.bert(input_ids_1, attention_mask_1)['pooler_output'], \
            self.bert(input_ids_2, attention_mask_2)['pooler_output']
        shared_index = 0
        individual_index = 1
        subset_index = 0
        # print("number of individual layers: ", len(self.individual))
        for val in self.connected_info:
            if not self.subsets:
                if val == "S":
                    pooled1 = self.shared[shared_index](pooled1)
                    pooled1 = nn.ReLU()(pooled1)
                    pooled2 = self.shared[shared_index](pooled2)
                    pooled2 = nn.ReLU()(pooled2)
                    shared_index += 1
                if val == "I":
                    pooled1 = self.individual[individual_index](pooled1)
                    pooled1 = nn.ReLU()(pooled1)
                    pooled2 = self.individual[individual_index](pooled2)
                    pooled2 = nn.ReLU()(pooled2)
                    individual_index += 3
            else: 
                if val == "S" and self.connected_info[0] != 1:
                    pooled1 = self.shared[shared_index](pooled1)
                    pooled1 = nn.ReLU()(pooled1)
                    pooled2 = self.shared[shared_index](pooled2)
                    pooled2 = nn.ReLU()(pooled2)
                    shared_index += 1
                if val == "S" and self.connected_info[0] == 1:
                    pooled1 = self.individual[-1 - subset_index](pooled1)
                    pooled1 = nn.ReLU()(pooled1)
                    pooled2 = self.individual[-1 - subset_index](pooled2)
                    pooled2 = nn.ReLU()(pooled2)
                    subset_index += 1
                if val == "I":
                    pooled1 = self.individual[individual_index](pooled1)
                    pooled1 = nn.ReLU()(pooled1)
                    pooled2 = self.individual[individual_index](pooled2)
                    pooled2 = nn.ReLU()(pooled2)
                    individual_index += 3

        logit = self.ppr_linear(self.ppr_dropout(pooled1 + pooled2))
        return logit


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should a sensical input to the loss function used to evaluate STS; i.e., MSE loss.
        '''
        pooled1, pooled2 = self.bert(input_ids_1, attention_mask_1)['pooler_output'], \
            self.bert(input_ids_2, attention_mask_2)['pooler_output']
        # transform pooled outputs: 
        shared_index = 0
        individual_index = 2
        subset_index = 0
        for val in self.connected_info:
            if not self.subsets:
                if val == "S":
                    pooled1 = self.shared[shared_index](pooled1)
                    pooled1 = nn.ReLU()(pooled1)
                    pooled2 = self.shared[shared_index](pooled2)
                    pooled2 = nn.ReLU()(pooled2)
                    shared_index += 1
                if val == "I":
                    pooled1 = self.individual[individual_index](pooled1)
                    pooled1 = nn.ReLU()(pooled1)
                    pooled2 = self.individual[individual_index](pooled2)
                    pooled2 = nn.ReLU()(pooled2)
                    individual_index += 3
            else: 
                print(val, individual_index)

                if val == "S" and self.connected_info[0] != 2:
                    pooled1 = self.shared[shared_index](pooled1)
                    pooled1 = nn.ReLU()(pooled1)
                    pooled2 = self.shared[shared_index](pooled2)
                    pooled2 = nn.ReLU()(pooled2)
                    shared_index += 1
                if val == "S" and self.connected_info[0] == 2:
                    pooled1 = self.individual[-1-subset_index](pooled1)
                    pooled1 = nn.ReLU()(pooled1)
                    pooled2 = self.individual[-1-subset_index](pooled2)
                    pooled2 = nn.ReLU()(pooled2)
                    subset_index += 1
                if val == "I":
                    pooled1 = self.individual[individual_index](pooled1)
                    pooled1 = nn.ReLU()(pooled1)
                    pooled2 = self.individual[individual_index](pooled2)
                    pooled2 = nn.ReLU()(pooled2)
                    individual_index += 3


        transform1, transform2 = self.sim_linear(self.sim_dropout(pooled1)), \
            self.sim_linear(self.sim_dropout(pooled2))
        # compute cosine similarity score:  
        logit = (F.cosine_similarity(transform1, transform2) + 1) * 2.5
        return logit


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def cycle(iterable): 
    """ Custom cycle generator which restarts an iterable after it is exhausted. """
    while True: 
        for elem in iterable: 
            yield elem 

def train_multitask(tune_config):
    args = tune_config['args']
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    
    # load data
    # create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(args.sst_train, args.para_train, args.sts_train, split ='train')
    sst_dev_data, num_labels, para_dev_data, sts_dev_data       = load_multitask_data(args.sst_dev,   args.para_dev,   args.sts_dev,   split ='train')

    # sst data: 
    sst_train_data       = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data         = SentenceClassificationDataset(sst_dev_data, args)
    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True,  batch_size=args.batch_size, collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader   = DataLoader(sst_dev_data,   shuffle=False, batch_size=args.batch_size, collate_fn=sst_dev_data.collate_fn)

    # paraphrase data: 
    para_train_data       = SentencePairDataset(para_train_data, args)
    para_dev_data         = SentencePairDataset(para_dev_data, args)
    para_train_dataloader = DataLoader(para_train_data, shuffle=True,  batch_size=args.batch_size, collate_fn=para_train_data.collate_fn)
    para_dev_dataloader   = DataLoader(para_dev_data,   shuffle=False, batch_size=args.batch_size, collate_fn=para_dev_data.collate_fn)
    
    # semantic similarity data: 
    sts_train_data       = SentencePairDataset(sts_train_data, args)
    sts_dev_data         = SentencePairDataset(sts_dev_data,   args)
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True,  batch_size=args.batch_size, collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader   = DataLoader(sts_dev_data,   shuffle=False, batch_size=args.batch_size, collate_fn=sts_dev_data.collate_fn)
    
    # cycle generators for each dataloader: 
    para_dataiter, sst_dataiter, sts_dataiter = \
        iter(cycle(para_train_dataloader)), iter(cycle(sst_train_dataloader)), iter(cycle(sts_train_dataloader))
    
    # init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}
    config = SimpleNamespace(**config)

    model = MultitaskBERT(config, tune_config['connected'], tune_config['subsets'])
    model = model.to(device)

    lr = tune_config['lr']
    [para_lossweight, sst_lossweight, sts_lossweight] = tune_config['lossweights']
    optimizer = AdamW(model.parameters(), lr=lr)
    best_multitask_score = 0
    best_model = model
        
    # run for the specified number of epochs: 
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0    
        
        # iterate through training set dataloader with maximum number of batches (NUM_BATCHES_PER_EPOCH): 
        for _ in tqdm(range(NUM_BATCHES_PER_EPOCH), desc=f'training epoch num. {epoch}', disable=TQDM_DISABLE): 
        
        # CURRENTLY NOT IN USE:
        # for step, batch in enumerate(tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
            # kill training if limiting paraphrase train batches and limit batch # hit: 
            # if LIMIT_PARAPHRASE_TRAIN_BATCHES: 
            #     if step == PARAPHRASE_MAX_BATCHES: break 
                
            # compute loss for sst task: 
            sst_batch = next(sst_dataiter)     
            
            b_ids, b_mask, b_labels = (sst_batch['token_ids'], sst_batch['attention_mask'], sst_batch['labels'])
            b_ids, b_mask, b_labels = b_ids.to(device), b_mask.to(device), b_labels.to(device)
            
            logits = model.predict_sentiment(b_ids, b_mask)
            sst_loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size            
            
            # compute loss for para task: 
            para_batch = next(para_dataiter)
            
            (b_ids1, b_mask1, b_ids2, b_mask2, b_labels) = \
                (para_batch['token_ids_1'], para_batch['attention_mask_1'], para_batch['token_ids_2'], para_batch['attention_mask_2'], para_batch['labels'])
            b_ids1, b_mask1, b_ids2, b_mask2, b_labels = \
                b_ids1.to(device), b_mask1.to(device), b_ids2.to(device), b_mask2.to(device), b_labels.to(device).to(torch.float32)
                
            logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
            y_hat = logits.sigmoid().round().flatten()
            b_labels = b_labels.flatten()
            para_loss = F.mse_loss(y_hat, b_labels.view(-1), reduction='sum') / args.batch_size
            
            # compute loss for sts task: 
            sts_batch = next(sts_dataiter)
            
            (b_ids1, b_mask1, b_ids2, b_mask2, b_labels) = \
                (sts_batch['token_ids_1'], sts_batch['attention_mask_1'], sts_batch['token_ids_2'], sts_batch['attention_mask_2'], sts_batch['labels'])
            b_ids1, b_mask1, b_ids2, b_mask2, b_labels = \
                b_ids1.to(device), b_mask1.to(device), b_ids2.to(device), b_mask2.to(device), b_labels.to(device).to(torch.float32)

            logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
            y_hat = logits.flatten()
            b_labels = b_labels.flatten()
            sts_loss = F.mse_loss(y_hat, b_labels.view(-1), reduction='sum') / args.batch_size
            
            # compute average loss and propagate loss: 
            optimizer.zero_grad()
            """ NOTE TO SELF: UNCOMMENT-COMMENT FOR RUNS USING VARIABLE LOSS WEIGHTS"""
            loss = (sst_loss * sst_lossweight) + (para_loss * para_lossweight) + (sts_loss * sts_lossweight)
            # loss = (sst_loss + para_loss + sts_loss) / 3
            loss = loss.type(torch.cuda.FloatTensor)
            loss.backward()
            optimizer.step()
            
            # record total training loss and batch count: 
            train_loss  += loss.item()
            num_batches += 1
        
        # evaluate models and report training characteristics: 
        train_loss = train_loss / (num_batches)

        # training sets evaluation: 
        print("\nTRAINING SETS EVALUATIONS...")
        """ NOTE TO SELF: UNCOMMENT FOR SINGLETON RUNS """
        (train_para_accuracy, _, _, train_sentiment_accuracy, _, _, train_sts_corr, _, _) = \
            model_eval_train_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, device)
        
        # dev sets evaluation: 
        print("\nDEV SETS EVALUATIONS...")
        (dev_paraphrase_accuracy, _ , _ , dev_sentiment_accuracy, _ , _ , dev_sts_corr, _ , _) = \
            model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)
        
        # if there is a multitask score improvement: 
        multitask_score = (dev_paraphrase_accuracy + dev_sentiment_accuracy + dev_sts_corr) / 3
        if multitask_score > best_multitask_score: 
            best_multitask_score = multitask_score
            """ NOTE TO SELF: UNCOMMENT ALL LINES FOR SINGLETON RUNS"""
            # save the new best model: 
            print("\nSAVING MODEL...")
            save_model(model, optimizer, args, config, args.filepath)
            # locally retain the best model: 
            best_model = deepcopy(model)

        # report training loss
        """ NOTE TO SELF: UNCOMMENT FOR SINGLETON RUNS"""
        print(f"\nEPOCH {epoch}: TRAINING LOSS :: {train_loss :.3f}; DEV MULTITASK SCORE :: {multitask_score :.3f}")
        
        # log training characterisitcs (training set loss & score characteristics per epoch)
        """ NOTE TO SELF: UNCOMMENT FOR SINGLETON RUNS"""
        with open(ABSOLUTE_PATH_PREFIX + "training_characteristics.txt", 'a+', encoding='utf-8') as f: 
            for line in f:
                if line.isspace(): break
            f.write(f'Epoch {epoch}, training loss :: {train_loss :.3f}, training set task scores [para. acc. - senti. acc. - sts. corr.] :: {train_para_accuracy :.3f}, {train_sentiment_accuracy :.3f}, {train_sts_corr :.3f}, dev set task scores [para. acc. - senti. acc. - sts. corr.] :: {dev_paraphrase_accuracy :.3f}, {dev_sentiment_accuracy :.3f}, {dev_sts_corr :.3f}, dev set mtt score: {multitask_score :.3f}' + '\n')
    
    # CURRENTLY NOT IN USE:    
    # print("FIRST SET DEV MODEL MULTITASK FIGS...")
    # (dev_paraphrase_accuracy, _ , _ , dev_sentiment_accuracy, _ , _ , dev_sts_corr, _ , _) = \
    #         model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)
    # print(f'dev sent acc :: {dev_sentiment_accuracy :.3f}, dev para acc :: {dev_paraphrase_accuracy :.3f}, dev sts corr :: {dev_sts_corr :.3f}')
    # print("SECOND SET DEV MODEL MULTITASK FIGS...")
    # test_model(args, tune_config=tune_config)
    
    # test the model within this training function: 
    """ NOTE TO SELF: UNCOMMENT FOR SINGLETON RUNS"""
    test_model_multitask(args, best_model, device)
    print("DONE TESTING BEST MODEL...")
    
    return {"score": best_multitask_score}
        

def test_model(args, tune_config):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config, tune_config)
        model.load_state_dict(saved, strict=False)
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train",  type=str, default=ABSOLUTE_PATH_PREFIX + "data/ids-sst-train.csv")
    parser.add_argument("--sst_dev",    type=str, default=ABSOLUTE_PATH_PREFIX + "data/ids-sst-dev.csv")
    parser.add_argument("--sst_test",   type=str, default=ABSOLUTE_PATH_PREFIX + "data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default=ABSOLUTE_PATH_PREFIX + "data/quora-train.csv")
    parser.add_argument("--para_dev",   type=str, default=ABSOLUTE_PATH_PREFIX + "data/quora-dev.csv")
    parser.add_argument("--para_test",  type=str, default=ABSOLUTE_PATH_PREFIX + "data/quora-test-student.csv")

    parser.add_argument("--sts_train",  type=str, default=ABSOLUTE_PATH_PREFIX + "data/sts-train.csv")
    parser.add_argument("--sts_dev",    type=str, default=ABSOLUTE_PATH_PREFIX + "data/sts-dev.csv")
    parser.add_argument("--sts_test",   type=str, default=ABSOLUTE_PATH_PREFIX + "data/sts-test-student.csv")

    parser.add_argument("--seed",   type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out",  type=str, default=ABSOLUTE_PATH_PREFIX + "predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default=ABSOLUTE_PATH_PREFIX + "predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out",  type=str, default=ABSOLUTE_PATH_PREFIX + "predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default=ABSOLUTE_PATH_PREFIX + "predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out",  type=str, default=ABSOLUTE_PATH_PREFIX + "predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default=ABSOLUTE_PATH_PREFIX + "predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)

    args = parser.parse_args()
    return args


def tune_train(args):
    ray.init(include_dashboard=False, log_to_driver=True, num_gpus=1) # , object_store_memory=1* 1024 * 1024 * 1024)
    config = {
        "connected": tune.grid_search([
            # ["S", "S", "S"],
            #["I", "S", "S"],
            # ["I", "I", "S"],
            # ["I", "I", "I"],
            # ["I", "S", "I"],
            # ["S", "I", "S"],
            # ["S", "I", "I"],
            # ["S", "S", "I"], 
            [0, "I", "S", "S"],
            [1, "I", "S", "S"],
            [2, "I", "S", "S"],
            [0, "S", "S", "I"],
            [1, "S", "S", "I"],
            [2, "S", "S", "I"],
            [0, "S", "I", "S"],
            [1, "S", "I", "S"],
            [2, "S", "I", "S"],

        ]),

        "subsets": True, 
        
        
        "lossweights" : tune.grid_search([
            [0.33, 0.33, 0.33], # default // no changes val. (order - para, sst, sts)
        #     # [0.40, 0.40, 0.20], 
        #     [0.40, 0.20, 0.40], # try this one 
        #     # [0.20, 0.40, 0.40], 
        #     # [0.50, 0.25, 0.25],
        #     # [0.25, 0.50, 0.25], # and this one
        #     # [0.25, 0.25, 0.50]
        ]), 
        "lr" : 8.56487702692805E-06,
        # "lr" : tune.loguniform(1e-1, 1e-7, base=10),
        "args": args,                              
    }
    # tuner = tune.Tuner(train_multitask, param_space=config)
    tuner = tune.Tuner(train_multitask, param_space=config, tune_config=tune.TuneConfig(metric='score', mode='max', num_samples=1, max_concurrent_trials=1))
    results = tuner.fit()
    print("\nBEST RESULTS CONFIG...")
    best_result_config = results.get_best_result(metric="score", mode="max").config
    print(' REGIME: ' + str(best_result_config['connected']) + ' LEARNING RATE: ' + str(best_result_config['lr']) + \
        ' LOSS WEIGHTING (PARA, SST, STS): ' + str(best_result_config['lossweights']))
    
    # print(results.get_dataframe())
    # print(str(results.get_best_result(metric="score", mode="max").config['connected']))
    # print(str(results.get_best_result(metric="score", mode="max").config['lr']))
    
    return results.get_best_result(metric="score", mode="max").config


if __name__ == "__main__":    
    args = get_args()
    args.filepath = ABSOLUTE_PATH_PREFIX + f'{args.option}-epochs_{args.epochs}-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    best_config = tune_train(args)
    # # currently tesitng the model within the tune_train() function call:
    # uncomment this line if testing outside of tune_train(): test_model(args, best_config)

# before each run, remember to check/move: 
    # training_characteristics.txt file 
    # predictions from /predictions dir 
    # .pt model file 
# from /minbert-default-final-project directory to ./extensions_materials 