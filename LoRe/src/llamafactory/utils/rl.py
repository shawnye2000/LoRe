import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn



class PolicyNet(torch.nn.Module):
    def __init__(self, categories_len, provider_num):
        super(PolicyNet, self).__init__()
        self.categories_len = categories_len
        self.provider_num = provider_num
        state_dim = self.provider_num * (self.categories_len + 1)
        hidden_dim = state_dim * 2
        action_dim = self.provider_num*(self.categories_len + 1)
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 2 * hidden_dim)
        self.fc3 = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, action_dim)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x):
        batch_size = x.shape[0]
        # print(f'xxxxxxx:{x}')
        # print(f'xxxxxxx:{x}')
        x_trans = x.reshape(batch_size, self.provider_num*(self.categories_len + 1)).float()
        x1 = F.relu(self.fc1(x_trans))
        # print(f'x1111111:{x1}')
        x2 = F.relu(self.fc2(x1))
        # print(f'x2222222:{x2}')
        x3 = F.relu(self.fc3(x2))
        y = self.fc4(x3).reshape(batch_size, self.provider_num, self.categories_len + 1)
        # print(f'yyyyyyy:{y}')

        return F.softmax(y, dim=-1)


class ValueNet(torch.nn.Module):
    def __init__(self, categories_len, provider_num):
        super(ValueNet, self).__init__()
        self.provider_num = provider_num
        self.categories_len = categories_len
        state_dim = self.provider_num * (self.categories_len + 1)
        hidden_dim = state_dim * 2
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 2 * hidden_dim)
        self.fc3 = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, state_dim)
        self.fc5 = torch.nn.Linear(state_dim, 1)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.xavier_uniform_(self.fc5.weight)

    def forward(self, x):
        batch_size = x.shape[0]
        x_trans = x.reshape(batch_size, self.provider_num*(self.categories_len + 1)).float()
        x1 = F.relu(self.fc1(x_trans))
        x2 = F.relu(self.fc2(x1))
        x3 = F.relu(self.fc3(x2))
        x4 = F.relu(self.fc4(x3))
        return self.fc5(x4)

class PolicyNet_user(torch.nn.Module):
    def __init__(self, config, cates_num):
        super(PolicyNet_user, self).__init__()
        self.config = config
        # embedding_size = self.config["embedding_size"]
        n_users = self.config['agent_num'] + 1
        torch.manual_seed(self.config['seed'])
        self.user_embedding = nn.Embedding(n_users, self.config["user_id_dim"], padding_idx=0)
        self.item_embedding = nn.Embedding(self.config['max_item_num'], self.config["item_id_dim"], padding_idx=0)
        self.cate_embedding = nn.Embedding(cates_num + 1, self.config["item_cate_id_dim"], padding_idx=0)
        self.item_feat_size = self.config["item_id_dim"] + self.config["item_cate_id_dim"]
        self.attn = AttentionSequencePoolingLayer(embedding_dim=self.item_feat_size)
        self.total_dim_of_all_fileds = self.config["user_id_dim"] + self.config["item_id_dim"] * 2 + self.config["item_cate_id_dim"] * 2
        self.fc_layer = FullyConnectedLayer(input_size=self.total_dim_of_all_fileds,
                                            hidden_unit=self.config['hid_units'],
                                            batch_norm=False,
                                            sigmoid = True,
                                            activation='relu',
                                            dropout=self.config['dropout'],
                                            dice_dim=2)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.cate_embedding.weight)
        self.user_embedding.weight.data[0] = torch.zeros(self.config["user_id_dim"])
        self.item_embedding.weight.data[0] = torch.zeros(self.config["item_id_dim"])
        self.cate_embedding.weight.data[0] = torch.zeros(self.config["item_cate_id_dim"])

    def forward(self, user, items, items_cate, interest_items, interest_items_cate):
        # print(f'user:{user.shape}') # [1]
        # print(f'items:{items.shape}') # [1, 18]
        # print(f'interest_items:{interest_items.shape}') # [1, 5]
        # user_embedding = self.user_embedding(user).unsqueeze(1)
        # item_embedding = self.item_embedding(items)
        # interest_items_embedding = self.item_embedding(interest_items).view(item_embedding.shape[0], 1, -1)

        # user_feature = torch.cat([interest_items_embedding, user_embedding], dim = -1)
        # user_feature = user_feature.expand(-1, item_embedding.shape[1], -1)
        # concat_feature = torch.cat([item_embedding, user_feature], dim = -1)
        # # print(f'concat_feature:{concat_feature}')
        # output = self.fc_layer(concat_feature).squeeze(-1)
        # output = torch.where(items == 0, torch.tensor(-float('inf')), output)
        # # print(f'output:{output}')
        # return F.softmax(output, dim=-1)
        user_emb, item_emb, rec_his_emb, mask = self.embed_input_fields_for_attention_pooling(user, items, items_cate, interest_items, interest_items_cate)

        browse_atten = self.attn(item_emb.unsqueeze(dim=2),
                            rec_his_emb, mask)
        concat_feature = torch.cat([item_emb, browse_atten.squeeze(dim=2), user_emb], dim=-1) # [18, 160]

        output = self.fc_layer(concat_feature).squeeze(dim=-1)
        output = torch.where(items == 0, torch.tensor(-float('inf')), output)

        return F.softmax(output, dim=-1)
    
    def embed_input_fields_for_attention_pooling(self, user, items, items_cate, interest_items, interest_items_cate):
        # def mean_pooling(tensor_data, id_data, dim):
        #     mask = id_data != 0
        #     tensor_data = torch.sum(tensor_data * mask.unsqueeze(-1), dim=dim)
        #     tensor_data = tensor_data / (torch.sum(mask, dim=dim, keepdim=True) + 1e-9)  # 1e-9 to avoid dividing zero

        #     return tensor_data
        
        user_emb = self.user_embedding(user)

        item_emb = self.item_embedding(items)
        # if len(items.size()) == 2:
        #     item_emb = mean_pooling(item_emb, items, 1)
        items_cate_emb = self.cate_embedding(items_cate)
        item_emb_ls = [item_emb, items_cate_emb]

        interest_item_emb = self.item_embedding(interest_items)
        # if len(interest_items.size()) == 3:
        #     interest_item_emb = mean_pooling(interest_item_emb, interest_items, 2)
        interest_items_cate_emb = self.cate_embedding(interest_items_cate)
        history_emb_ls = [interest_item_emb, interest_items_cate_emb]
        
        history_mask = interest_items == 0.0

        return user_emb, torch.cat(item_emb_ls, dim=-1), torch.cat(history_emb_ls, dim=-1), history_mask

class ValueNet_user(torch.nn.Module):
    def __init__(self, config, cates_num):
        super(ValueNet_user, self).__init__()
        self.config = config
        embedding_size = self.config["embedding_size"]
        n_users = self.config['agent_num'] + 1
        torch.manual_seed(self.config['seed'])
        self.user_embedding = nn.Embedding(n_users, self.config["user_id_dim"], padding_idx=0)
        self.item_embedding = nn.Embedding(self.config['max_item_num'], self.config["item_id_dim"], padding_idx=0)
        self.cate_embedding = nn.Embedding(cates_num + 1, self.config["item_cate_id_dim"], padding_idx=0)
        self.item_feat_size = self.config["item_id_dim"] + self.config["item_cate_id_dim"]
        self.attn = AttentionSequencePoolingLayer(embedding_dim=self.item_feat_size)
        self.total_dim_of_all_fileds = self.config["user_id_dim"] + self.config["item_id_dim"] * 2 + self.config["item_cate_id_dim"] * 2
        self.fc_layer = FullyConnectedLayer(input_size=self.total_dim_of_all_fileds,
                                            hidden_unit=self.config['hid_units'],
                                            batch_norm=False,
                                            sigmoid = False,
                                            activation='relu',
                                            dropout=self.config['dropout'],
                                            dice_dim=2)
        # self.out_layer = torch.nn.Linear(state_dim, 1)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.cate_embedding.weight)
        self.user_embedding.weight.data[0] = torch.zeros(self.config["user_id_dim"])
        self.item_embedding.weight.data[0] = torch.zeros(self.config["item_id_dim"])
        self.cate_embedding.weight.data[0] = torch.zeros(self.config["item_cate_id_dim"])

    def forward(self, user, items, items_cate, interest_items, interest_items_cate):

        user_emb, item_emb, rec_his_emb, mask = self.embed_input_fields_for_attention_pooling(user, items, items_cate, interest_items, interest_items_cate)

        browse_atten = self.attn(item_emb.unsqueeze(dim=2),
                            rec_his_emb, mask)
        concat_feature = torch.cat([item_emb, browse_atten.squeeze(dim=2), user_emb], dim=-1) # [18, 160]

        mid_res = self.fc_layer(concat_feature).squeeze(dim=-1)
        # output = torch.where(items == 0, 0, output)
        output = torch.mean(mid_res, dim=1).unsqueeze(-1)
        # print(f'predict reward:{output}')
        return output
        # user_embedding = self.user_embedding(user).unsqueeze(1)
        # item_embedding = self.item_embedding(items)
        # item_cate_embedding = self.cate_embedding(items_cate)
        # interest_items_embedding = self.item_embedding(interest_items).view(interest_items.shape[0], 1, -1)
        # interest_items_cate_embedding = self.cate_embedding(interest_items_cate).view(interest_items_cate.shape[0], 1, -1)
        
        # user_feature = torch.cat([interest_items_cate_embedding, interest_items_embedding, user_embedding], dim = -1)
        # user_feature = user_feature.expand(-1, item_embedding.shape[1], -1)
        # concat_feature = torch.cat([item_embedding, user_feature], dim = -1)
        
        # mid_feature = self.fc_layer(concat_feature).squeeze(-1)
        # output = torch.where(items == 0, 0.0, mid_feature)

        # # user_embedding = self.user_embedding(user).squeeze(1)
        # # item_embedding = self.item_embedding(items)
        # # interest_items_embedding = F.relu(self.fc(self.item_embedding(interest_items))).sum(1)
        # # user_feature = user_embedding + interest_items_embedding
        # # user_feature = user_feature.unsqueeze(1)
        
        # # mul = F.relu(self.fc_out(user_feature * item_embedding)).squeeze(-1)
        # return output.sum(-1)

    def embed_input_fields_for_attention_pooling(self, user, items, items_cate, interest_items, interest_items_cate):
        # def mean_pooling(tensor_data, id_data, dim):
        #     mask = id_data != 0
        #     tensor_data = torch.sum(tensor_data * mask.unsqueeze(-1), dim=dim)
        #     tensor_data = tensor_data / (torch.sum(mask, dim=dim, keepdim=True) + 1e-9)  # 1e-9 to avoid dividing zero

        #     return tensor_data
        
        user_emb = self.user_embedding(user)

        item_emb = self.item_embedding(items)
        # if len(items.size()) == 2:
        #     item_emb = mean_pooling(item_emb, items, 1)
        items_cate_emb = self.cate_embedding(items_cate)
        item_emb_ls = [item_emb, items_cate_emb]

        interest_item_emb = self.item_embedding(interest_items)
        # if len(interest_items.size()) == 3:
        #     interest_item_emb = mean_pooling(interest_item_emb, interest_items, 2)
        interest_items_cate_emb = self.cate_embedding(interest_items_cate)
        history_emb_ls = [interest_item_emb, interest_items_cate_emb]

        history_mask = interest_items == 0.0
        
        return user_emb, torch.cat(item_emb_ls, dim=-1), torch.cat(history_emb_ls, dim=-1), history_mask

class FullyConnectedLayer(nn.Module):
    def __init__(self, input_size, hidden_unit, batch_norm=False, activation='relu', sigmoid=False, dropout=None, dice_dim=None):
        super(FullyConnectedLayer, self).__init__()
        assert len(hidden_unit) >= 1 
        self.sigmoid = sigmoid
        layers = []
        layers.append(nn.Linear(input_size, hidden_unit[0]))
        
        for i, h in enumerate(hidden_unit[:-1]):
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_unit[i]))
            
            if activation.lower() == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation.lower() == 'tanh':
                layers.append(nn.Tanh())
            elif activation.lower() == 'leakyrelu':
                layers.append(nn.LeakyReLU())
            # elif activation.lower() == 'dice':
            #     assert dice_dim
            #     layers.append(Dice(hidden_unit[i], dim=dice_dim))
            else:
                raise NotImplementedError

            if dropout is not None:
                layers.append(nn.Dropout(dropout))
            
            layers.append(nn.Linear(hidden_unit[i], hidden_unit[i+1]))

        self.fc = nn.Sequential(*layers)
        if self.sigmoid:
            self.output_layer = nn.Sigmoid()
        

    def forward(self, x):
        return self.output_layer(self.fc(x)) if self.sigmoid else self.fc(x)
    
class AttentionSequencePoolingLayer(nn.Module):
    def __init__(self, embedding_dim=4):
        super(AttentionSequencePoolingLayer, self).__init__()

        self.local_att = LocalActivationUnit(hidden_unit=[64, 16], embedding_dim=embedding_dim, batch_norm=False)

    
    def forward(self, query_ad, user_behavior, mask=None):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size
        # mask                : size -> batch_size * time_seq_len
        # output              : size -> batch_size * 1 * embedding_size
        
        attention_score = self.local_att(query_ad, user_behavior)
        attention_score = torch.transpose(attention_score, 2, 3)  # [1, 18, 1, 5] # mask: [1, 18, 5]

        if mask is not None:
            attention_score = attention_score.masked_fill(mask.unsqueeze(2), torch.tensor(0))
        

        # multiply weight
        output = torch.matmul(attention_score, user_behavior)

        return output
        

class LocalActivationUnit(nn.Module):
    def __init__(self, hidden_unit=[80, 40], embedding_dim=4, batch_norm=False):
        super(LocalActivationUnit, self).__init__()
        self.fc1 = FullyConnectedLayer(input_size=4*embedding_dim,
                                       hidden_unit=hidden_unit,
                                       batch_norm=batch_norm,
                                       sigmoid=False,
                                       activation='relu',
                                       dice_dim=3)

        self.fc2 = nn.Linear(hidden_unit[-1], 1)

    # @torchsnooper.snoop()
    def forward(self, query, user_behavior):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size

        user_behavior_len = user_behavior.size(2)
        
        queries = query.expand(-1, -1, user_behavior_len, -1)
        
        attention_input = torch.cat([queries, user_behavior, queries-user_behavior, queries*user_behavior],
             dim=-1) # as the source code, subtraction simulates verctors' difference
        
        attention_output = self.fc1(attention_input)
        attention_score = self.fc2(attention_output) # [B, T, 1]

        return attention_score