import torch
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from torch.nn import functional as F
from recommender_system.preference_estimator import user_preference_estimator


class MeLU(torch.nn.Module):
    def __init__(self, config):
        super(MeLU, self).__init__()
        self.use_cuda = config['use_cuda']
        self.model = user_preference_estimator(config)
        self.local_lr = config['local_lr']
        self.store_parameters()
        self.theta = []
        for i in range(2653): # Task specific learning rates
            self.theta.append(self.local_lr)
        self.meta_lr = config['meta_lr']
        self.meta_optim = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        self.task_specific_learner = torch.optim.Adam(self.model.parameters(), lr=config['meta_lr'])
        self.local_update_target_weight_name = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'linear_out.weight',
                                                'linear_out.bias']

    def store_parameters(self):
        self.keep_weight = deepcopy(self.model.state_dict())
        self.weight_name = list(self.keep_weight.keys())
        self.weight_len = len(self.keep_weight)
        self.fast_weights = OrderedDict()

    def forward(self, support_set_x, support_set_y, query_set_x, num_local_update, task):
        for idx in range(num_local_update):
            if idx > 0:
                self.model.load_state_dict(self.fast_weights)
            weight_for_local_update = list(self.model.state_dict().values())
            support_set_y_pred = self.model(support_set_x)
            loss = F.mse_loss(support_set_y_pred, support_set_y.view(-1, 1))
            self.model.zero_grad()
            grad = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            # local update
            for i in range(self.weight_len):
                if self.weight_name[i] in self.local_update_target_weight_name:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i] - self.theta[task] * grad[i]
                else:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i]
        self.model.load_state_dict(self.fast_weights)
        query_set_y_pred = self.model(query_set_x)
        self.model.load_state_dict(self.keep_weight)
        return query_set_y_pred

    def global_update(self, support_set_xs, support_set_ys, query_set_xs, query_set_ys, num_local_update):
        batch_sz = len(support_set_xs)
        losses_q = []
        training_loss_pre = []
        training_loss_after = []
        if self.use_cuda:
            for i in range(batch_sz):
                support_set_xs[i] = support_set_xs[i].cuda()
                support_set_ys[i] = support_set_ys[i].cuda()
                query_set_xs[i] = query_set_xs[i].cuda()
                query_set_ys[i] = query_set_ys[i].cuda()
        for i in range(batch_sz):  # All the tasks within the batch are trained one-by-one
            training_loss_pre.append(F.mse_loss(self.model(query_set_xs[i]), query_set_ys[i].view(-1, 1)).item())

            query_set_y_pred = self.forward(support_set_xs[i], support_set_ys[i], query_set_xs[i], num_local_update, 1)
            loss_q = F.mse_loss(query_set_y_pred, query_set_ys[i].view(-1, 1))
            losses_q.append(loss_q)  # The training loss for each task is stored

            training_loss_after.append(loss_q.item())

        losses_q = torch.stack(losses_q).mean(0)  # This is regarded as the mean loss for the batch.
        self.meta_optim.zero_grad()
        losses_q.backward()
        self.meta_optim.step()
        self.store_parameters()

        pre_loss = np.array(training_loss_pre)
        after_loss = np.array(training_loss_after)
        train_loss = np.mean(pre_loss)
        valid_loss = np.mean(after_loss)

        return train_loss, valid_loss
