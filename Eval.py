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





def tensorify(lst):
    """
    List must be nested list of tensors (with no varying lengths within a dimension).
    Nested list of nested lengths [D1, D2, ... DN] -> tensor([D1, D2, ..., DN)

    :return: nested list D
    """
    # base case, if the current list is not nested anymore, make it into tensor
    if type(lst[0]) != list:
        if type(lst) == torch.Tensor:
            return lst
        elif type(lst[0]) == torch.Tensor:
            return torch.stack(lst, dim=0)
        else:  # if the elements of lst are floats or something like that
            return torch.tensor(lst)
    current_dimension_i = len(lst)
    for d_i in range(current_dimension_i):
        tensor = tensorify(lst[d_i])
        lst[d_i] = tensor
    # end of loop lst[d_i] = tensor([D_i, ... D_0])
    tensor_lst = torch.stack(lst, dim=0)
    return tensor_lst




class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        self.temp = None

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)



import pickle

train_loader_list = []
val_loader_list = []

with open(project_path+'Pickles/Domain/pickle_train_en_fb_yt.pickle', 'rb') as f:
    td = pickle.load(f)
    train_loader_list.append(td)

with open(project_path+'Pickles/Domain/pickle_train_en_twitter.pickle', 'rb') as f:
    td = pickle.load(f)
    train_loader_list.append(td)

with open(project_path+'Pickles/Domain/pickle_train_en_wiki.pickle', 'rb') as f:
    td = pickle.load(f)
    train_loader_list.append(td)


with open(project_path+'Pickles/Domain/pickle_train_en_stormfront.pickle', 'rb') as f:
    test_loader = pickle.load(f)

with open(project_path+'Pickles/Domain/pickle_train_en_stormfront.pickle', 'rb') as f:
    fine_loader = pickle.load(f)



zero_shot_list = ['stormfront', 'fox', 'twi_fb', 'reddit', 'convAI', 'HateCheck','gab', 'yt_reddit',]

for d in zero_shot_list:
    with open(project_path+'Pickles/Domain/pickle_train_en_'+str(d)+'.pickle', 'rb') as f:
        td = pickle.load(f)
        val_loader_list.append(td)


from torch.utils.data import DataLoader

for i in range(len(train_loader_list)):
  train_loader_list[i] = DataLoader(train_loader_list[i], batch_size=8, shuffle=True)

test_loader = DataLoader(test_loader, batch_size=8)
fine_loader = DataLoader(fine_loader, batch_size=8)


for i in range(len(val_loader_list)):
  val_loader_list[i] = DataLoader(val_loader_list[i], batch_size=8, shuffle=False)




with open(project_path+'Pickles/Domain/pickle_train_en_ERM_yt_twi_wiki.pickle', 'rb') as f:
    erm_train_loader = pickle.load(f)
erm_train_loader = DataLoader(erm_train_loader, batch_size=8, shuffle=True)



#model path
erm_path = "Saved_Models/en_bert_erm.pt"
fish_path = "Saved_Models/en_bert_fish.pt"
scl_fish_path = "Saved_Models/en_bert_scl_fish.pt"
scl_erm_path = "Saved_Models/en_bert_scl_erm.pt"





class BertClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        #self.post_init()

    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))
    
    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        labels = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        return_logits=False,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if(return_logits):
          return logits, pooled_output
        return logits





def test(test_loader, loader_type='test', verbose=True, save_ypred=False, num_labels = 2):
    model.eval()
    yhats, ys, metas = [], [], []
    with torch.no_grad():
        for i, b in enumerate(test_loader):
            # get the inputs
            x_i, x_a, y = b['input_ids'].to(device), b['attention_mask'].to(device), b['labels'].to(device)
            y_hat = model(input_ids = x_i, attention_mask = x_a, labels = y)
            ys.append(y)
            yhats.append(y_hat)

        preds = torch.cat(yhats)
        _, predicted = torch.max(preds.data, 1)
        ypreds =  predicted
        ys = torch.cat(ys)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(preds.view(-1, num_labels), ys.view(-1))
        accuracy = accuracy_score(ys.cpu(), ypreds.cpu())
        bin_f1 = f1_score(ys.cpu(), ypreds.cpu(), average='binary')
        macro_f1 = f1_score(ys.cpu(), ypreds.cpu(), average='macro')

    return loss.item(), accuracy, bin_f1, macro_f1





print('===== ZERO-SHOT ============')

model = BertClassifier.from_pretrained('BertForMaskedLM', num_labels=2).to(device)


#############################################

print('======= ERM =============')

PATH = erm_path
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])

avg_acc_list = []
avg_bin_f1_list = []
avg_macro_f1_list = []

for i, lang in enumerate(zero_shot_list):
    print('________  '+str(lang)+' ___________')
    loss_test, accuracy, bin_f1, macro_f1 = test(val_loader_list[i], loader_type='val', verbose=False)
    print('Acc: '+ str(accuracy))
    print('Binary F1: '+ str(bin_f1))
    print('Macro F1: '+ str(macro_f1))
    avg_acc_list.append(accuracy)
    avg_bin_f1_list.append(bin_f1)
    avg_macro_f1_list.append(macro_f1)

print('__________ AVG. Accuracy ___________' +str(sum(avg_acc_list)/len(avg_acc_list)))
print('__________ AVG. Binary F1 ___________' +str(sum(avg_bin_f1_list)/len(avg_bin_f1_list)))
print('__________ AVG. Macro F1 ___________' +str(sum(avg_macro_f1_list)/len(avg_macro_f1_list)))
    
###################################################################################################

print('======= FISH =============')

PATH = fish_path
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])

avg_acc_list = []
avg_bin_f1_list = []
avg_macro_f1_list = []

for i, lang in enumerate(zero_shot_list):
    print('________  '+str(lang)+' ___________')
    loss_test, accuracy, bin_f1, macro_f1 = test(val_loader_list[i], loader_type='val', verbose=False)
    print('Acc: '+ str(accuracy))
    print('Binary F1: '+ str(bin_f1))
    print('Macro F1: '+ str(macro_f1))
    avg_acc_list.append(accuracy)
    avg_bin_f1_list.append(bin_f1)
    avg_macro_f1_list.append(macro_f1)

print('__________ AVG. Accuracy ___________' +str(sum(avg_acc_list)/len(avg_acc_list)))
print('__________ AVG. Binary F1 ___________' +str(sum(avg_bin_f1_list)/len(avg_bin_f1_list)))
print('__________ AVG. Macro F1 ___________' +str(sum(avg_macro_f1_list)/len(avg_macro_f1_list)))
    
###################################################################################################


print('======= SCL FISH =============')

PATH = scl_fish_path
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])

avg_acc_list = []
avg_bin_f1_list = []
avg_macro_f1_list = []

for i, lang in enumerate(zero_shot_list):
    print('________  '+str(lang)+' ___________')
    loss_test, accuracy, bin_f1, macro_f1 = test(val_loader_list[i], loader_type='val', verbose=False)
    print('Acc: '+ str(accuracy))
    print('Binary F1: '+ str(bin_f1))
    print('Macro F1: '+ str(macro_f1))
    avg_acc_list.append(accuracy)
    avg_bin_f1_list.append(bin_f1)
    avg_macro_f1_list.append(macro_f1)

print('__________ AVG. Accuracy ___________' +str(sum(avg_acc_list)/len(avg_acc_list)))
print('__________ AVG. Binary F1 ___________' +str(sum(avg_bin_f1_list)/len(avg_bin_f1_list)))
print('__________ AVG. Macro F1 ___________' +str(sum(avg_macro_f1_list)/len(avg_macro_f1_list)))
    
###################################################################################################


print('======= SCL ERM =============')

PATH = scl_erm_path
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])

avg_acc_list = []
avg_bin_f1_list = []
avg_macro_f1_list = []

for i, lang in enumerate(zero_shot_list):
    print('________  '+str(lang)+' ___________')
    loss_test, accuracy, bin_f1, macro_f1 = test(val_loader_list[i], loader_type='val', verbose=False)
    print('Acc: '+ str(accuracy))
    print('Binary F1: '+ str(bin_f1))
    print('Macro F1: '+ str(macro_f1))
    avg_acc_list.append(accuracy)
    avg_bin_f1_list.append(bin_f1)
    avg_macro_f1_list.append(macro_f1)

print('__________ AVG. Accuracy ___________' +str(sum(avg_acc_list)/len(avg_acc_list)))
print('__________ AVG. Binary F1 ___________' +str(sum(avg_bin_f1_list)/len(avg_bin_f1_list)))
print('__________ AVG. Macro F1 ___________' +str(sum(avg_macro_f1_list)/len(avg_macro_f1_list)))

    
###################################################################################################




print('Done !!!!!!!!!!!!!!!!!')

