import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.data_utils import one_hot
from methods.meta_template import MetaTemplate
import wandb
import torch.nn.functional as F

from typing import List, Literal, Callable

from backbones.blocks import full_block

from enum import Enum

class DistanceType(Enum):
    """ Enum for the different distance types. """
    FC_CONC = 'fc-conc'
    FC_DIFF = 'fc-diff'
    EUCLIDEAN = 'euclidean'
    COSINE = 'cosine'
    L1 = 'l1'

class RelationModule(nn.Module):
    """ Relation module for RelationNet. """
    def __init__(self,
                 feat_dim: int,
                 deep_distance_type: Literal['l1', 'euclidean', 'cosine', 'fc-conc', 'fc-diff'],
                 deep_distance_layer_sizes: List[int],
                 dropout: float
        ):
        super(RelationModule, self).__init__()
        
        self.relation_module = nn.Sequential()
        in_size = feat_dim * (2 if deep_distance_type == 'fc-conc' else 1)
        for i, out_size in enumerate(deep_distance_layer_sizes[:-1]):
            # Add dropout to all layers except the last one
            self.relation_module.add_module(f'fb{i}', full_block(in_size, out_size, dropout=dropout))
            in_size = out_size
            
        self.relation_module.add_module('classifier', nn.Linear(in_size, deep_distance_layer_sizes[-1]))
        self.relation_module.add_module('sigmoid', nn.Sigmoid())

    def forward(self, x):
        return self.relation_module(x)

class RelationNet(MetaTemplate):
    def __init__(
            self,
            backbone: nn.Module,
            n_way: int,
            n_support: int,
            deep_distance_layer_sizes: List[int] = [128, 64, 64, 32, 32, 8, 1],
            deep_distance_type: Literal['l1', 'euclidean', 'cosine', 'fc-conc', 'fc-diff'] = 'l1',
            representative_aggregation: Callable[[torch.Tensor, int], torch.Tensor] | Callable[[torch.Tensor], torch.Tensor] = torch.mean,
            learning_rate: float = 1e-4,
            backbone_weight_decay: float = 1e-5,
            relation_module_dropout: float = 0.0,
        ):
        super(RelationNet, self).__init__(backbone, n_way, n_support)

        self.loss_fn = nn.MSELoss().cuda()
        self.distance_type = deep_distance_type
        self.representative_aggregation = representative_aggregation
        print(f"The representative aggregation is {self.representative_aggregation}")
        # Define the relation module, either as deep distance or as a simple distance for ablation
        self.relation_module = self.get_relation_module(deep_distance_layer_sizes, deep_distance_type, relation_module_dropout)
        
        self.backbone_optim = torch.optim.Adam(backbone.parameters(), lr=learning_rate, weight_decay=backbone_weight_decay)
        self.relation_module_optim = None
        if deep_distance_type in ['fc-conc', 'fc-diff']:
            self.relation_module_optim = torch.optim.Adam(self.relation_module.parameters(), lr=learning_rate)

    def get_relation_module(
            self, 
            deep_distance_layer_sizes: List[int], 
            deep_distance_type: Literal['l1', 'euclidean', 'cosine', 'fc-conc'], 
            dropout: float
        ):
        """ Creates the relation module, either as deep distance or as a simple distance for ablation. """
        relation_module = None
        match deep_distance_type:
            case 'fc-conc' | 'fc-diff':
                relation_module = RelationModule(self.feat_dim, deep_distance_type, deep_distance_layer_sizes, dropout).cuda()
            case 'euclidean':
                relation_module = lambda x: -F.pairwise_distance(x[:, :x.shape[1]//2], x[:, x.shape[1]//2:], p=2).reshape(-1, 1)
            case 'cosine':
                relation_module = lambda x: (1 + F.cosine_similarity(x[:, :x.shape[1]//2], x[:, x.shape[1]//2:], dim=1).reshape(-1, 1)) / 2 # Map the cosine similarity [-1, 1] to [0, 1]
            case 'l1':
                relation_module = lambda x: -F.pairwise_distance(x[:, :x.shape[1]//2], x[:, x.shape[1]//2:], p=1).reshape(-1, 1)
            case _: 
                raise ValueError('deep_distance_type must be one of [fc, euclidean, cosine, l1]')
            
        return relation_module

    def set_forward(self, x, is_feature=False):
        """ Forward pass for a single few-shot episode.
        Args:
            x: Variable containing support and query sets of shape (n_way, n_support + n_query, data_dim)
            is_feature: True if x is already a feature tensor, False otherwise

        Returns:
            relation_scores: Tensor containing the relation scores of shape (n_way * n_query, n_way)
        """
        # Get support and query set embeddings of shape (n_way, n_support, feat_dim) and (n_way * n_query, feat_dim)
        z_support, z_query = self.parse_feature(x, is_feature)
        z_support = z_support.contiguous()
        z_query = z_query.contiguous().reshape(-1, self.feat_dim)

        # Element-wise sum/mean over support examples for each class to form a single class representative, of shape (n_way, feat_dim)
        z_support = self.representative_aggregation(z_support, 1) # torch.sum(z_support, 1)

        # Expand the class representatives to match the number of queries, of shape (n_way**2 * n_query, feat_dim)
        z_support_ext = z_support.unsqueeze(0).repeat(self.n_way * self.n_query, 1, 1).view(-1, self.feat_dim)
        
        # Expand the query embeddings contiguously to match the number of classes, of shape (n_way**2 * n_query, feat_dim)
        z_query_ext = z_query.repeat_interleave(self.n_way, 0)

        # Combine support and query embeddings to form pairs, of shape (n_way**2 * n_query, feat_dim * 2) for concatenation and (n_way**2 * n_query, feat_dim) for difference
        if self.distance_type == 'fc-diff':
            relation_pairs = z_support_ext - z_query_ext
        else:
            relation_pairs = torch.cat((z_support_ext, z_query_ext), 1)
        
        # Pass the pairs through the relation network, of shape (n_way * n_query, n_way)
        relation_scores = self.relation_module(relation_pairs).view(-1, self.n_way)

        if self.distance_type in ['euclidean', 'l1']:
            relation_scores = F.softmax(relation_scores, dim=1) # Apply softmax to the distances to get a probability distribution

        return relation_scores
        
    def set_forward_loss(self, x):
        """ Loss for a single few-shot episode.
        Args:
            x: Variable containing support and query sets of shape (n_way, n_support + n_query, data_dim)

        Returns:
            loss: The MSE loss for the episode
        """
        # Vector of the form [0, ..., 0, 1, ..., 1, ..., n_way, ..., n_way] of shape (n_way * n_query)
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        # Transform it as a one-hot vector of shape (n_way * n_query, n_way)
        y_query = Variable(one_hot(y_query, self.n_way).cuda())

        # Compute the relation scores for the episode and compute the MSE loss
        relation_scores = self.set_forward(x)
        return self.loss_fn(relation_scores, y_query)
    
    def train_loop(self, epoch, train_loader, optimizer, print_freq=10):
        """ Training loop for RelationNet.
        Args:
            epoch: Current epoch
            train_loader: Dataloader for training episodes
            optimizer: Optimizer to use for training (Not used in RelationNet, as it uses two optimizers)
        """
        #wandb.watch(self, log="parameters", log_freq=print_freq)

        avg_loss = 0
        for i, (x, _) in enumerate(train_loader):
            # x is a tuple of support and query sets, each of shape (n_way, n_support + n_query, data_dim)
            if isinstance(x, list):
                self.n_query = x[0].size(1) - self.n_support
                if self.change_way:
                    self.n_way = x[0].size(0)
            else: 
                self.n_query = x.size(1) - self.n_support
                if self.change_way:
                    self.n_way = x.size(0)

            # Compute loss for a single episode and perform a gradient step
            self.backbone_optim.zero_grad()
            if self.relation_module_optim is not None:
                self.relation_module_optim.zero_grad()

            loss = self.set_forward_loss(x)
            loss.backward()

            self.backbone_optim.step()
            if self.relation_module_optim is not None:
                self.relation_module_optim.step()

            avg_loss = avg_loss + loss.item()

            # Log the loss
            if i % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss / float(i + 1)))
                wandb.log({"loss": avg_loss / float(i + 1)})

    def correct(self, x):
        scores = self.set_forward(x)
        y_query = np.repeat(range(self.n_way), self.n_query)

        #print('scores', scores.shape, scores)
        #print('y_query', y_query.shape, y_query)

        _, topk_labels = scores.data.topk(1, 1, True, True)
        #print('topk_labels', topk_labels.shape, topk_labels)
        topk_ind = topk_labels.cpu().numpy()
        #print('topk_ind', topk_ind.shape, topk_ind)
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        #print('top1_correct', top1_correct.shape, top1_correct)
        return float(top1_correct), len(y_query)

    def test_loop(self, test_loader, return_std=False):
        """ Testing loop for RelationNet.
        Args:
            test_loader: Dataloader for testing episodes
            return_std: True if the standard deviation should be returned, False otherwise

        Returns:
            acc_mean: Mean accuracy over all episodes
            acc_std: Standard deviation over all episodes
        """
        acc_all = []
        iter_num = len(test_loader)
        for x, _ in test_loader:
            # x is a tuple of support and query sets, each of shape (n_way, n_support + n_query, data_dim)
            if isinstance(x, list):
                self.n_query = x[0].size(1) - self.n_support
                if self.change_way:
                    self.n_way = x[0].size(0)
            else: 
                self.n_query = x.size(1) - self.n_support
                if self.change_way:
                    self.n_way = x.size(0)

            # Compute the accuracy for a single episode
            #print('x', x.shape, x)
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this / count_this * 100)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)

        # Log the accuracy
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        
        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean