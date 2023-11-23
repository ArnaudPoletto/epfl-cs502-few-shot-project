import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
from utils.data_utils import one_hot
from methods.meta_template import MetaTemplate

class RelationNet(MetaTemplate):
    def __init__(self, backbone, n_way, n_support, feature_dim=64, relation_dim=8):
        super(RelationNet, self).__init__(backbone, n_way, n_support)

        self.loss_fn = nn.MSELoss().cuda()
        
        self.relation_module = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.feat_dim * 2, 64)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(64, 64)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(64, 1)),
            ('sigmoid', nn.Sigmoid())
        ])).cuda()

    def set_forward(self, x, is_feature=False):
        """ Forward pass for a single few-shot episode.
        Args:
            x: Variable containing support and query sets of shape (n_way, n_support + n_query, data_dim)
            is_feature: True if x is already a feature tensor, False otherwise

        Returns:
            TODO
        """
        #print(f"x: {x.shape} {x}")

        # Get support and query set embeddings of shape (n_way, n_support, feat_dim) and (n_way * n_query, feat_dim)
        z_support, z_query = self.parse_feature(x, is_feature)
        z_support = z_support.contiguous()
        z_query = z_query.contiguous().view(self.n_way * self.n_query, self.feat_dim)
        #print(f"z_support: {z_support.shape}")
        #print(f"z_query: {z_query.shape}")

        # Element-wise sum over support examples for each class to form a single class representative, of shape (n_way, feat_dim)
        z_support = torch.sum(z_support, 1)
        #print(f"z_support after sum: {z_support.shape}")

        # Expand the class representatives to match the number of queries, of shape (n_way * n_query, n_way, feat_dim)
        z_support_ext = z_support.unsqueeze(0).repeat(self.n_way * self.n_query, 1, 1)
        # Reshape to (n_way**2 * n_query, feat_dim)
        z_support_ext = z_support_ext.view(-1, self.feat_dim)
        #print(f"z_support_ext: {z_support_ext.shape}")
        
        # Expand the query embeddings to match the number of classes, of shape (n_way, n_way * n_query, feat_dim)
        z_query_ext = z_query.unsqueeze(0).repeat(self.n_way, 1, 1)
        # Reshape to (n_way**2 * n_query, feat_dim)
        z_query_ext = z_query_ext.view(-1, self.feat_dim)
        #print(f"z_query_ext: {z_query_ext.shape}")

        # Combine support and query embeddings to form pairs, of shape (n_way, n_way * n_query, feat_dim * 2)
        relation_pairs = torch.cat((z_support_ext, z_query_ext), 1)
        #print(f"relation_pairs: {relation_pairs.shape}")
        
        # Pass the pairs through the relation network, of shape (n_way * n_query, n_way)
        relation_scores = self.relation_module(relation_pairs).view(-1, self.n_way)
        #print(f"relation_scores: {relation_scores.shape} {relation_scores}")
        
        return relation_scores
        
    def set_forward_loss(self, x):
        """ Loss for a single few-shot episode.
        Args:
            x: Variable containing support and query sets of shape (n_way, n_support + n_query, data_dim)

        Returns:
            TODO
        """
        # Vector of the form [0, ..., 0, 1, ..., 1, ..., n_way, ..., n_way] of shape (n_way * n_query)
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        # Transform it as a one-hot vector of shape (n_way * n_query, n_way)
        y_query = Variable(one_hot(y_query, self.n_way).cuda())

        relation_scores = self.set_forward(x)
        #print(f"relation_scores: {relation_scores.shape} {relation_scores}")
        #print(f"y_query: {y_query.shape} {y_query}")

        #print(f"relation_scores: {relation_scores.shape}")
        #print(f"y_query: {y_query.shape}")
        
        return self.loss_fn(relation_scores, y_query)
        #print(f"loss: {loss.shape}")
        
        return loss