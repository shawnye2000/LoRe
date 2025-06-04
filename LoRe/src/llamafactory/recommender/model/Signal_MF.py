import importlib
from typing import Union, List, Optional
from .base import BaseModel
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Signal_MF(BaseModel, nn.Module):
    def __init__(self, config):
        BaseModel.__init__(self, config)
        nn.Module.__init__(self)
        self.config = config
        self.embedding_size = config["embedding_size"]
        self.n_users = config['agent_num'] + 1
        # self.n_items = n_items
        torch.manual_seed(config['seed'])
        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.config['max_item_num'], self.embedding_size)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, user, item):
        """Predicts the rating of a user for an item."""
        user_embed = self.user_embedding(user)
        item_embed = self.item_embedding(item)  # 新的item 没办法初始化


        # Dot product between user and item embeddings to predict rating
        predicted_rating = (user_embed * item_embed).sum(1)
        predicted_rating = F.sigmoid(predicted_rating)
        return predicted_rating

    def get_pro(self, user, item):
        predicted_ratings = self.forward(user, item)
        predicted_ratings = torch.clamp(predicted_ratings, min = 1e-10, max = 1.0)
        return predicted_ratings

    def get_full_sort_items(self, user, items):
        """Get a list of sorted items for a given user."""
        predicted_ratings = self.forward(user, items)
        print(f'actor_pro:{predicted_ratings}')
        predicted_ratings = F.softmax(predicted_ratings, dim = 0)
        sorted_items = self._sort_full_items(predicted_ratings, items)
        return sorted_items.tolist(), predicted_ratings.tolist()

    def _sort_full_items(self, predicted_ratings, items):
        """Sort items based on their predicted ratings."""
        # Sort items based on ratings in descending order and return item indices
        _, sorted_indices = torch.sort(predicted_ratings, descending=True)
        return items[sorted_indices]
