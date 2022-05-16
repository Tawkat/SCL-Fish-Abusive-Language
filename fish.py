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







class ParamDict(OrderedDict):
    """A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)


def fish_step(meta_weights, inner_weights, meta_lr):
    meta_weights, weights = ParamDict(meta_weights), ParamDict(inner_weights)
    meta_weights += meta_lr * sum([weights - meta_weights], 0 * meta_weights)
    return meta_weights






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





def fish(train_loader_list, num_labels, meta_lr, num_epochs):
    model.train()

    Best_loss = 1000
    Best_loss_test = 10000
    Best_f1 = -1
    Best_acc = -1
    Best_epoch = -1
    pretrain_iters = 100
    i = 0

    Best_Test_acc = -1
    Best_Test_f1 = -1
    

    for epoch in range(num_epochs):
        print('\n====> Epoch: {:03d} '.format(epoch))
        opt_inner_pre = None

        # inner loop update
        for b0, b1, b2 in list(itertools.zip_longest(train_loader_list[0], train_loader_list[1], train_loader_list[2])):
            num_none_domain = 0
            i+=1

            model_inner = copy.deepcopy(model)
            model_inner.train()
            opt_inner = Adam(model_inner.parameters(), lr = 5e-6)
            if opt_inner_pre is not None:
                opt_inner.load_state_dict(opt_inner_pre)

            #for ind, b in enumerate(train_loader):
            if(b0 is not None):
                # get the inputs
                opt_inner.zero_grad()
                x_i, x_a, y = b0['input_ids'].to(device), b0['attention_mask'].to(device), b0['labels'].to(device)
                y_hat = model_inner(input_ids = x_i, attention_mask = x_a, labels = y)
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(y_hat.view(-1, num_labels), y.view(-1))
                loss.backward()
                opt_inner.step()
            else:
                num_none_domain += 1

            #for ind, b in enumerate(train_loader):
            # get the inputs
            #i+=1
            if(b1 is not None):
                opt_inner.zero_grad()
                x_i, x_a, y = b1['input_ids'].to(device), b1['attention_mask'].to(device), b1['labels'].to(device)
                y_hat = model_inner(input_ids = x_i, attention_mask = x_a, labels = y)
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(y_hat.view(-1, num_labels), y.view(-1))
                loss.backward()
                opt_inner.step()
            else:
                num_none_domain += 1

            if(num_none_domain == len(train_loader_list)-1): #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                break

            #for ind, b in enumerate(train_loader):
            # get the inputs
            #i+=1
            if(b2 is not None):
                opt_inner.zero_grad()
                x_i, x_a, y = b2['input_ids'].to(device), b2['attention_mask'].to(device), b2['labels'].to(device)
                y_hat = model_inner(input_ids = x_i, attention_mask = x_a, labels = y)
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(y_hat.view(-1, num_labels), y.view(-1))
                loss.backward()
                opt_inner.step()
            else:
                num_none_domain += 1


            opt_inner_pre = opt_inner.state_dict()
            # fish update
            meta_weights = fish_step(meta_weights=model.state_dict(),
                                      inner_weights=model_inner.state_dict(),
                                      meta_lr=meta_lr / len(train_loader_list))
            model.reset_weights(meta_weights)
            

            if (i + 1) % pretrain_iters == 0:
                loss_test, accuracy, f1 = test(test_loader, loader_type='val', verbose=False)
                print('Loss Test ', loss_test)
                print('Accuracy ', accuracy)
                print('F1 Score ', f1)

                if((loss_test<Best_loss_test) and (f1>Best_Test_f1)): ################################
                    Best_loss_test = loss_test
                    Best_Test_acc = accuracy
                    Best_Test_f1 = f1

                    PATH = fish_path
                    torch.save({
                          'model_state_dict': model.state_dict(),
                          }, PATH)


                model.train()


    print('Best Test Loss ', Best_loss_test)
    print('TEST Accuracy: ', Best_Test_acc)
    print('TEST F1 Score ', Best_Test_f1)
    
    return Best_loss, Best_acc, Best_f1





model = BertClassifier.from_pretrained('BertForMaskedLM', num_labels=2).to(device)
optimiserC = Adam(model.parameters(), lr = 5e-6)

loss, acc, f1 = train_fish(train_loader_list, 2, meta_lr=0.05, num_epochs=10)



print('Finished FISH training!')
print('loss ', loss)
print('Best Acc ', acc)
print('F1 Score ', f1)



print('Done !!!!!!!!!!!!!!!!!')

