import torch

import numpy as np
import pandas as pd



import copy
from copy import deepcopy
import itertools
import operator
from numbers import Number
from collections import OrderedDict
import argparse

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


def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="BertForMaskedLM", help="the pre-trained model name")
    parser.add_argument("--temperature", type=float, default=0.05, help='temperature for SCL')
    parser.add_argument("--number_of_gpu", type=int, default=1, help="Number of available GPUs.")
    parser.add_argument("--save_model", type=str, default="Saved_Models/en_bert_scl_erm.pt", help="Directory to save model")
    parser.add_argument("--batch_size_train", type=int, default=8, help='batch size for in-platform dataset.')
    parser.add_argument("--batch_size_SCL", type=int, default=12, help='batch size for SCL.')
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=10)


    return parser.parse_args()

args = parse_config()


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
  train_loader_list[i] = DataLoader(train_loader_list[i], batch_size=args.batch_size_train, shuffle=True)

test_loader = DataLoader(test_loader, batch_size=args.batch_size_train)
fine_loader = DataLoader(fine_loader, batch_size=args.batch_size_train)


for i in range(len(val_loader_list)):
  val_loader_list[i] = DataLoader(val_loader_list[i], batch_size=args.batch_size_train, shuffle=False)




with open(project_path+'Pickles/Domain/pickle_train_en_ERM_yt_twi_wiki.pickle', 'rb') as f:
    erm_train_loader = pickle.load(f)
erm_train_loader = DataLoader(erm_train_loader, batch_size=args.batch_size_train, shuffle=True)


'''
#model path
erm_path = "Saved_Models/en_bert_erm.pt"
fish_path = "Saved_Models/en_bert_fish.pt"
scl_fish_path = "Saved_Models/en_bert_scl_fish.pt"
scl_erm_path = "Saved_Models/en_bert_scl_erm.pt"
'''





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
        f1 = f1_score(ys.cpu(), ypreds.cpu())

    return loss.item(), accuracy, f1





class SupConLossPLMS(torch.nn.Module):

    def __init__(self, device, temperature=0.05):
        super(SupConLossPLMS, self).__init__()
        self.tem = temperature
        self.device = device

    def forward(self, batch_emb, labels=None):
        labels = labels.view(-1, 1)
        batch_size = batch_emb.shape[0]
        #print(batch_size)
        mask = torch.eq(labels, labels.T).float()
        norm_emb = F.normalize(batch_emb, dim=1, p=2)
        # compute logits
        dot_contrast = torch.div(torch.matmul(norm_emb, norm_emb.T), self.tem)
        # for numerical stability
        logits_max, _ = torch.max(dot_contrast, dim=1, keepdim=True,)  
        logits = dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(self.device), 0,)
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        #print(exp_logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        #print(log_prob)
        mask_sum = mask.sum(1)
        #print(mask_sum)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        #print(mask_sum)
        mean_log_prob_pos = -(mask * log_prob).sum(1) / mask_sum
        #print(mean_log_prob_pos)
        return mean_log_prob_pos.mean()






def SCL_ERM(train_loaders, num_labels = 2, pretrain_iters=100, num_epochs=5):

    n_iters = 0
    Best_loss = 10000
    Best_loss_test = 10000
    Best_acc = -1
    Best_f1 = -1
    i = 0

    Best_Test_acc = -1
    Best_Test_f1 = -1
    
    for epoch in range(num_epochs):
        
        for b0 in train_loaders:            
            i+=1
            x_i_all, x_a_all, y_all = [], [], []
            model.train()
            # get the inputs
 
            optimiserC.zero_grad()
            x_i, x_a, y = b0['input_ids'].to(device), b0['attention_mask'].to(device), b0['labels'].to(device)
            y_hat = model(input_ids = x_i, attention_mask = x_a, labels = y)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(y_hat.view(-1, num_labels), y.view(-1))
            loss.backward()
            optimiserC.step()

            #n_iters += 1
            # display progress

            x_i_all.append(copy.deepcopy(x_i))
            x_a_all.append(copy.deepcopy(x_a))
            y_all.append(copy.deepcopy(y))
        



            x_i = torch.cat(x_i_all)
            x_a = torch.cat(x_a_all)
            y = torch.cat(y_all)
            dataset_cl = TensorDataset(x_i, x_a, y)
            dataloader_cl = DataLoader(dataset_cl, batch_size=args.batch_size_SCL, shuffle=True)
            
            model.train()

            for dt in dataloader_cl:
                optimiserC.zero_grad()
                b_i, b_a, y = dt[0].to(device), dt[1].to(device), dt[2].to(device)
                # get the inputs
                y_hat, logits = model(input_ids = b_i, attention_mask = b_a, labels = y, return_logits=True)

                loss_scl_criterion = SupConLossPLMS(device = device, temperature=args.temperature)
                loss_scl = loss_scl_criterion(logits, y.view(-1))
                loss_scl.backward()
                optimiserC.step()

            
            n_iters += 1
            # display progress
            if (i + 1) % pretrain_iters == 0:
                loss_test, accuracy, f1 = test(test_loader, loader_type='val', verbose=False)
                print('Loss Test ', loss_test)
                print('Accuracy ', accuracy)
                print('F1 Score ', f1)

                if((loss_test<Best_loss_test) and (f1>Best_Test_f1)):
                    Best_loss_test = loss_test
                    Best_Test_acc = accuracy
                    Best_Test_f1 = f1


                    PATH = args.save_model
                    torch.save({
                          'model_state_dict': model.state_dict(),
                          }, PATH)


    #pbar.close()

    print('Finished SCL_ERM pre-training!')
    print('Accuracy ', Best_acc)
    print('F1 Score ', Best_f1)
    print('Best Test Loss ', Best_loss_test)
    print('TEST Accuracy: ', Best_Test_acc)
    print('TEST F1 Score ', Best_Test_f1)





model = BertClassifier.from_pretrained(args.model_name, num_labels=args.num_labels).to(device)
optimiserC = Adam(model.parameters(), lr = args.learning_rate)

SCL_ERM(erm_train_loader, pretrain_iters=100, num_epochs=args.num_epochs)



print('Done !!!!!!!!!!!!!!!!!')

