import torch
import numpy as np
from copy import deepcopy

from torch.autograd import Variable
from torch.nn import functional as F
from collections import OrderedDict
from options import config

from embeddings import item, user


class user_preference_estimator(torch.nn.Module):
    def __init__(self, config):
        super(user_preference_estimator, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.fc1_in_dim = config['embedding_dim'] * 8
        self.fc2_in_dim = config['first_fc_hidden_dim']
        self.fc2_out_dim = config['second_fc_hidden_dim']
        self.use_cuda = config['use_cuda']

        self.item_emb = item(config)
        self.user_emb = user(config)
        self.fc1 = torch.nn.Linear(self.fc1_in_dim, self.fc2_in_dim)
        self.fc2 = torch.nn.Linear(self.fc2_in_dim, self.fc2_out_dim)
        self.linear_out = torch.nn.Linear(self.fc2_out_dim, 1)

    def forward(self, x, training = True):
        rate_idx = Variable(x[:, 0], requires_grad=False)
        genre_idx = Variable(x[:, 1:26], requires_grad=False)
        director_idx = Variable(x[:, 26:2212], requires_grad=False)
        actor_idx = Variable(x[:, 2212:10242], requires_grad=False)
        gender_idx = Variable(x[:, 10242], requires_grad=False)
        age_idx = Variable(x[:, 10243], requires_grad=False)
        occupation_idx = Variable(x[:, 10244], requires_grad=False)
        area_idx = Variable(x[:, 10245], requires_grad=False)

        item_emb = self.item_emb(rate_idx, genre_idx, director_idx, actor_idx)
        user_emb = self.user_emb(gender_idx, age_idx, occupation_idx, area_idx)
        x = torch.cat((item_emb, user_emb), 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.linear_out(x)


class MeLU(torch.nn.Module):
    def __init__(self, config):
        super(MeLU, self).__init__()
        self.use_cuda = config['use_cuda']
        self.model = user_preference_estimator(config)
        self.grads = [0 for _ in range(len(list(self.model.parameters())))]
        self.local_lr = config['local_lr']
        self.store_parameters()
        self.local_update_target_weight_name = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'linear_out.weight', 'linear_out.bias']

    def store_parameters(self):
        self.keep_weight = deepcopy(self.model.state_dict())
        self.weight_name = list(self.keep_weight.keys())
        self.weight_len = len(self.keep_weight)
        self.fast_weights = OrderedDict()


    def global_update(self, support_set_xs, support_set_ys, query_set_xs, query_set_ys, num_local_update, update=True):
        batch_sz = len(support_set_xs)
        losses_q = []
        self.model.cuda()
        
        for j in range(self.weight_len):
            self.grads[j] = 0
        
        for k in range(batch_sz):
            sset_x = support_set_xs[k].cuda()
            sset_y = support_set_ys[k].cuda()
            qset_x = query_set_xs[k].cuda()
            qset_y = query_set_ys[k].cuda()
            for idx in range(num_local_update):
                if idx > 0:
                    self.model.load_state_dict(self.fast_weights)
                self.model.train()
                weight_for_local_update = list(self.model.state_dict().values())
                sset_y_pred = self.model(sset_x)
                loss = F.mse_loss(sset_y_pred, sset_y.view(-1, 1))
                grad = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
                # local update
                for i in range(self.weight_len):
                    if self.weight_name[i] in self.local_update_target_weight_name:
                        self.fast_weights[self.weight_name[i]] = weight_for_local_update[i] - self.local_lr * grad[i]
                    else:
                        self.fast_weights[self.weight_name[i]] = weight_for_local_update[i]
            self.model.load_state_dict(self.fast_weights)
            query_set_y_pred = self.model(qset_x)
            
            if update:
                loss_q = F.mse_loss(query_set_y_pred, qset_y.view(-1, 1))
                grad = torch.autograd.grad(loss_q, self.model.parameters())
                for j in range(len(grad)):
                    self.grads[j] += grad[j] / batch_sz
                self.model.load_state_dict(self.keep_weight)
            else:
                losses_q.append(F.l1_loss(query_set_y_pred, qset_y.view(-1, 1)).item())
                self.model.load_state_dict(self.keep_weight)
        
        if update:
            self.meta_optim = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
            self.meta_optim.zero_grad()
            for i,j in enumerate(self.model.parameters()):
                j.grad = self.grads[i]
                j.grad.data.clamp_(-10, 10)
                
            self.meta_optim.step()
            self.store_parameters()
            return
        else:
            return losses_q

    def get_weight_avg_norm(self, support_set_x, support_set_y, num_local_update):
        tmp = 0.
        if self.cuda():
            support_set_x = support_set_x.cuda()
            support_set_y = support_set_y.cuda()
        for idx in range(num_local_update):
            if idx > 0:
                self.model.load_state_dict(self.fast_weights)
            weight_for_local_update = list(self.model.state_dict().values())
            support_set_y_pred = self.model(support_set_x)
            loss = F.mse_loss(support_set_y_pred, support_set_y.view(-1, 1))
            # unit loss
            loss /= torch.norm(loss).tolist()
            self.model.zero_grad()
            grad = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            for i in range(self.weight_len):
                # For averaging Forbenius norm.
                tmp += torch.norm(grad[i])
                if self.weight_name[i] in self.local_update_target_weight_name:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i] - self.local_lr * grad[i]
                else:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i]
        return tmp / num_local_update
