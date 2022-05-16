import torch

import numpy as np
import pandas as pd



import copy
from copy import deepcopy
import itertools
import operator
from numbers import Number
from collections import OrderedDict

from transformers import BertPreTrainedModel, BertModel
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader


from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score


seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)


# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")



project_path=''


model_name = 'BertFastTokenizer'
max_length = 512



class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 sentence,
                 label,
                 ):
        self.idx = idx
        self.sentence = sentence
        self.label = label



def read_examples(df,sentence_col='text',label_col='binary_labels'):
    """Read examples from filename."""
    examples=[]
    sentence_list = df[sentence_col].tolist()
    label_list = df[label_col].tolist()
    idx = 0

    assert len(sentence_list)==len(label_list)

    for i in range(len(sentence_list)):
        examples.append(
        Example(
                idx = idx,
                sentence=str(sentence_list[i]).strip(),
                label=label_list[i],
                ) 
        )
        idx+=1
    return examples


import logging

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 sentence_ids,
                 sentence_mask,
                 label,

    ):
        self.example_id = example_id
        self.sentence_ids = sentence_ids
        self.sentence_mask = sentence_mask
        self.label = label       
        


def convert_examples_to_features(examples, tokenizer, max_length = 512, padding='max_length',truncation=True,stage=None):
    features = []

    for example_index, example in enumerate(examples):
        sentence_tokenized = tokenizer(example.sentence, max_length=max_length,padding=padding,truncation=truncation)
        sentence_ids =  sentence_tokenized['input_ids']
        sentence_mask = sentence_tokenized['attention_mask']


        label = example.label


   
        if example_index < 3:
            #if stage=='train':
            logging.info("*** Example ***")
            logging.info("idx: {}".format(example.idx))

            logging.info("sentence_tokens: {}".format([x.replace('\u0120','_') for x in tokenizer.convert_ids_to_tokens(sentence_ids)]))
            logging.info("sentence_ids: {}".format(' '.join(map(str, sentence_ids))))
            logging.info("sentence_mask: {}".format(' '.join(map(str, sentence_mask))))
            
       
        features.append(
            InputFeatures(
                 example_index,
                 sentence_ids,
                 sentence_mask,
                 label,
            )
        )
    return features


import torch

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        self.temp = None

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)



from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

tokenizer = AutoTokenizer.from_pretrained(project_path + 'BertFastTokenizer') 


domain_list = ['wiki', 'fb_yt', 'twitter' , 'gab', 'convAI', 'fox', 'reddit', 'stormfront', 'twi_fb', 'HateCheck', 'yt_reddit']


train_features_all = []

for domain in domain_list:
  print(domain)
  train_file = project_path+'Datasets/combined_binary_en_'+str(domain)+'.csv'
  df = pd.read_csv(train_file)

  if(domain=='wiki'):
    df = df.sample(132815)

  #print(len(df['text']))
  #df = df.dropna()
  #print(len(df['text']))



  #df_train = df_train.sample(132815)
  print(len(df['binary_labels']))
  print(len(df[df['binary_labels']==0]), len(df[df['binary_labels']==1]))

  train_examples = read_examples(df)


  train_features = convert_examples_to_features(train_examples,tokenizer,max_length=max_length,padding='max_length',truncation=True,)


  train_dict = {}
  train_dict["input_ids"] = []
  train_dict["attention_mask"] = []
  train_labels = []

  for e in train_features:
    train_dict["input_ids"].append(e.sentence_ids)
    train_dict["attention_mask"].append(e.sentence_mask)
    train_labels.append(e.label)


  train_dataset = CustomDataset(train_dict,train_labels)



  with open(project_path+'Pickles/Domain/pickle_train_en_'+str(domain)+'.pickle', 'wb') as f:
    pickle.dump(train_dataset,f)

