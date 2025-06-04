import logging
import random

logging.basicConfig(level=logging.ERROR)
from datetime import datetime, timedelta, date
from typing import List
from termcolor import colored
import os
import logging
import argparse
from yacs.config import CfgNode
import csv
from tqdm import tqdm
import os
import time
import concurrent.futures
import json
from langchain.docstore import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS
from langchain_experimental.generative_agents import (
    GenerativeAgent,
    GenerativeAgentMemory,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import math
import faiss
import re
import dill
import numpy as np
import queue
from typing import List
import pandas as pd
import sys

import torch.nn as nn

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import sys
# import os

# print("Current working directory:", os.getcwd())  # 检查当前工作目录
# print("Python module search paths:")
# for path in sys.path:
#     print(path)
from llamafactory.agents import RecAgent, ProAgent
from llamafactory.recommender.recommender import Recommender
from llamafactory.recommender.data.data import Data
from llamafactory.utils import utils
from llamafactory.utils import utils, message
from llamafactory.utils.message import Message
from llamafactory.utils.event import Event, update_event, reset_event
from llamafactory.utils import interval as interval
from llamafactory.utils.rl import PolicyNet, ValueNet, PolicyNet_user, ValueNet_user
import threading
from llamafactory.agents.recagent_memory import RecAgentMemory, RecAgentRetriever
import heapq
from fastapi.middleware.cors import CORSMiddleware
from vllm import LLM as vvLLM
from vllm import SamplingParams
import transformers
lock = threading.Lock()
transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_seed(seed):
    '''setting random seeds'''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

setup_seed(2024)



class Simulator:
    """
    Simulator class for running the simulation.
    """

    def __init__(self, config: CfgNode, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.round_cnt = 0
        self.round_msg: List[Message] = []
        self.active_agents: List[int] = []  # active agents in current round
        self.active_proagents: List[int] = []
        self.active_agent_threshold = config["active_agent_threshold"]
        self.active_proagent_threshold = config["active_proagent_threshold"]
        self.active_method = config["active_method"]
        self.proagent_active_method = config["proagent_active_method"]
        self.file_name_path: List[str] = []
        self.play_event = threading.Event()
        self.working_agents: List[RecAgent] = []  # busy agents
        self.now = datetime.now().replace(hour=8, minute=0, second=0)
        self.interval = interval.parse_interval(config["interval"])
        self.round_entropy = []
        self.round_provider_num = []
        # self.rec_cnt = [20] * config["agent_num"]
        self.rec_stat = message.RecommenderStat(
            tot_user_num=0,
            cur_user_num=0,
            tot_item_num=0,
            inter_num=0,
            rec_model=config["rec_model"],
            pop_items=[],
        )


        self.new_round_item = []
        self.tokenizer, self.model = None, None
        self.embedding_size, self.embedding_model = utils.get_embedding_model()
        self.leave_providers = []

        # self.llm = vvLLM(model="/home/xiaopeng_ye/LLMs/Meta-Llama-3-8B-Instruct",
        #                  tensor_parallel_size=2,
        #                  # trust_remote_code=True,
        #                  )
        # self.tokenizer = AutoTokenizer.from_pretrained('/home/xiaopeng_ye/LLMs/Meta-Llama-3-8B-Instruct')
        if os.path.exists(self.config['profile_path']):
            with open(self.config['profile_path'], 'r') as f:
                self.provider_profile_dict = json.load(f)
        else:
            self.provider_profile_dict = {}

    def get_file_name_path(self):
        return self.file_name_path

    def load_simulator(self):
        """Load and initiate the simulator."""
        self.round_cnt = 0
        # self.embedding_model = utils.get_embedding_model()
        self.data = Data(self.config)
        self.agents = self.agent_creation()
        self.provider_agents, genre_list = self.provider_agent_creation()
        self.recsys = Recommender(self.config, self.logger, self.data)
        self.logger.info("Simulator loaded.")
        self.logger.info(f'Config :{self.config}')
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.categories = utils.get_item_categories(self.config['data_name'])
        self.genre_count = {}
        
        self.genre_popular = utils.get_genre_popular(self.config['data_name'])
        self.genre_click = {}

        # self.total_num = 0
        for gene in self.categories:
            self.genre_count[gene] = 0
            self.genre_click[gene] = 0
        # if not self.config['signal_user']:
        if not self.config['signal_user']:
            for uid, u_dict in self.data.users.items():
                for iid in u_dict['history']:
                    self.recsys.add_train_data(uid, iid, 1)
                all_item_id = list(self.data.items.keys())
                rest_item_id = [i for i in all_item_id if i not in u_dict['history']]
                for iid in random.sample(rest_item_id, 10):
                    self.recsys.add_train_data(uid, iid, 0)
        # else:
        #     #PPO: signal for users
        #     #self.temp_user_transition_dict[user] = {'user': None, 'interest_items': None, 'interest_items_cate': None, 'recent_items': None, 'recent_items_cate': None, 'actions': None, 'next_interest_items': None, 'next_interest_items_cate': None, 'rewards': 0}
        #     self.dict_count = []
        #     self.user_transition_dict = {'users': [], 'interest_items': [], 'interest_items_cate': [], 'recent_items': [], 'recent_items_cate': [], 'actions': [], 'next_interest_items': [], 'next_interest_items_cate': [], 'rewards': []}
        #     self.actor_user = PolicyNet_user(self.config, len(self.categories)).to(self.device)
        #     self.critic_user = ValueNet_user(self.config, len(self.categories)).to(self.device)
        #     if self.config['checkpoint']:
        #         self.actor_user.load_state_dict(torch.load(self.config['checkpoint_load_path'] + '/actor_user_200.pth'))
        #         self.critic_user.load_state_dict(torch.load(self.config['checkpoint_load_path'] + '/critic_user_200.pth'))
        #     self.actor_user_optimizer = torch.optim.Adam(self.actor_user.parameters(),
        #                                             lr=self.config['actor_user_lr'])
        #     self.critic_user_optimizer = torch.optim.Adam(self.critic_user.parameters(),
        #                                                             lr=self.config['critic_user_lr'])
        #     self.ppo_epochs_user = self.config['ppo_epochs_user']
        #     self.record = {}
        #     for user in self.data.get_full_users():
        #         self.record[user] = []
        #     self.interest_items = {}
        #     for user in self.data.get_full_users():
        #         self.interest_items[user] = [0] * self.config['Int_K']
        #     for uid, u_dict in self.data.users.items():
        #         for iid in u_dict['history']:
        #             self.interest_items[uid].insert(0, iid)
        #             self.interest_items[uid].pop()
        categories_len, provider_num = len(self.categories), len(self.provider_agents)
        self.action_dim = categories_len + 1
        if not self.config['signal']:
            return None
        
        #PPO: signal for provider
        
        self.actor = PolicyNet(categories_len, provider_num).to(self.device)
        self.critic = ValueNet(categories_len, provider_num).to(self.device)
        if self.config['ppo_load_checkpoint']:
            self.actor.load_state_dict(torch.load(self.config['ppo_checkpoint_load_path'] + '/actor_100.pth'))
            self.critic.load_state_dict(torch.load(self.config['ppo_checkpoint_load_path'] + '/critic_100.pth'))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.config['actor_lr'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                        lr=self.config['critic_lr'])
        self.gamma = self.config['gamma']
        self.lmbda = self.config['lmbda']
        self.ppo_epochs = self.config['ppo_epochs']
        self.eps = self.config['eps']

        genre_list_index = [self.categories.index(genre)+1  for genre in genre_list]
        state_one_hot = utils.int_to_onehot(genre_list_index, self.action_dim, self.device)
        init_state = state_one_hot.unsqueeze(0)
        return init_state

    def take_action(self, state):
        # state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        # print(f'probs:{probs}')
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.squeeze()
    
    def compute_advantage(self, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        advantage_array = np.array(advantage_list)
        return torch.tensor(advantage_array, dtype=torch.float)

    def update(self, transition_dict):
        states = torch.stack(transition_dict['states'], dim = 0).squeeze(1).float().to(self.device) # [batch_size, provider_len, categories_len+1]
        actions = torch.stack(transition_dict['actions'], dim = 0).unsqueeze(-1).to(self.device) # [batch_size, provider_len, 1]
        rewards = torch.tensor(transition_dict['rewards'], dtype = torch.float).view(-1, 1).to(self.device) # [batch_size, 1]
        next_states = torch.stack(transition_dict['next_states'], dim = 0).squeeze(1).float().to(self.device) # [batch_size, provider_len, categories_len+1]
        # dones = torch.tensor(transition_dict['dones'],
        #                      dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states)

        td_delta = td_target - self.critic(states)
        # exit()
        advantage = self.compute_advantage(td_delta.cpu()).to(self.device)
        # old_log_probs_all = torch.log(self.actor(states).gather(1, actions)).squeeze(-1).detach()
        # print(f'actions:{actions}')
        # print(f'self.actor(states):{self.actor(states)}')
        # print(f'self.actor(states).gather:{self.actor(states).gather(2, actions)}')
        old_log_probs_all = self.actor(states).gather(2, actions).detach()
        old_log_probs_all = torch.where(actions==0, torch.tensor(0, dtype=old_log_probs_all.dtype), old_log_probs_all).squeeze(-1)
        # print(f'states: {states.shape}') [2, 50, 16]
        # print(f'self.actor(states): {self.actor(states).shape}') [2, 50, 16]
        # print(f'actions: {actions.shape}') [2, 50, 1]
        # print(f'old_log_probs_all:{old_log_probs_all.shape}') [2, 50]
        # exit()
        old_log_probs = torch.sum(old_log_probs_all, dim=1)
        # print(f'td_target:{td_target.shape}')
        # print(f'td_delta:{td_delta.shape}')
        # print(f'advantage:{advantage}')
        # print(f'old_log_probs:{old_log_probs}')
        for _ in range(self.ppo_epochs):
            # log_probs_all = torch.log(self.actor(states).gather(1, actions)).squeeze(-1)
            # print(f'self.actor(states):{self.actor(states)}')
            # print(f'self.actor(states).gather:{self.actor(states).gather(2, actions)}')
            log_probs_all = self.actor(states).gather(2, actions)
            log_probs_all = torch.where(actions==0, torch.tensor(0, dtype=log_probs_all.dtype), log_probs_all).squeeze(-1)
            log_probs = torch.sum(log_probs_all, dim=1)
            # print(f'log_probs:{log_probs}')
            # print(f'log_probs:{log_probs.shape}')
            ratio = log_probs / old_log_probs


            # print(f'ratio:{ratio}')
            surr1 = ratio * advantage
            # print(f'surr1:{surr1.shape}')
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            # print(f'surr2:{surr2.shape}')
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            # print(f'advantage:{advantage}')
            # print(f'ratio:{ratio}')
            # print(f'surr2:{surr2}')
            # print(f'actor_loss:{actor_loss}')
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            # print(f'critic_loss:{critic_loss}')
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            # print('\n')

    def get_padded_data(self, input_data):
        users_input = input_data['users']
        interest_items_input = input_data['interest_items']
        interest_items_cate_input = input_data['interest_items_cate']
        recent_items_input = input_data['recent_items']
        recent_items_cate_input = input_data['recent_items_cate']
        actions_input = input_data['actions']
        next_interest_items_input = input_data['next_interest_items']
        next_interest_items_cate_input = input_data['next_interest_items_cate']
        rewards_input = input_data['rewards']

        max_len = max([tensor.size(0) for tensor in recent_items_input])
        padded_users = []
        padded_recent_items = []
        padded_recent_items_cate = []
        padded_interest_items = []
        padded_interest_items_cate = []
        padded_next_interest_items = []
        padded_next_interest_items_cate = []
        for user, items, items_cate, intere_items, intere_items_cate, next_intere_items, next_intere_items_cate in zip(users_input,\
                                           recent_items_input, recent_items_cate_input, \
                                           interest_items_input, interest_items_cate_input, \
                                            next_interest_items_input, next_interest_items_cate_input):
            if items.size(0) < max_len:
                # print(f'items.size(0):{items.size(0)}')
                # print(f'max_len:{max_len}')
                pad_len = max_len - items.size(0)
                # 填充 tensor
                padded_tensor = F.pad(items, (0, pad_len))  # padding (left, right)
                # print(f'items:{padded_tensor.shape}\n items:{padded_tensor}')
                padded_recent_items.append(padded_tensor)
                padded_tensor = F.pad(items_cate, (0, pad_len))
                # print(f'items_cate:{padded_tensor.shape}')
                padded_recent_items_cate.append(padded_tensor)
                padded_tensor = F.pad(user, (0, pad_len))
                # print(f'user:{padded_tensor.shape}\n user:{padded_tensor}')
                padded_users.append(padded_tensor)
                padded_tensor = F.pad(intere_items, (0, 0, 0, pad_len))
                # print(f'intere_items:{padded_tensor.shape}\n intere_items:{padded_tensor}')
                padded_interest_items.append(padded_tensor)
                padded_tensor = F.pad(intere_items_cate, (0, 0, 0, pad_len))
                # print(f'intere_items_cate:{padded_tensor.shape}')
                padded_interest_items_cate.append(padded_tensor)
                padded_tensor = F.pad(next_intere_items, (0, 0, 0, pad_len))
                # print(f'next_intere_items:{padded_tensor.shape}\n next_intere_items:{padded_tensor}')
                padded_next_interest_items.append(padded_tensor)
                padded_tensor = F.pad(next_intere_items_cate, (0, 0, 0, pad_len))
                # print(f'next_intere_items_cate:{padded_tensor.shape}')
                padded_next_interest_items_cate.append(padded_tensor)
                # exit()
            else:
                padded_recent_items.append(items)
                padded_recent_items_cate.append(items_cate)
                padded_users.append(user)
                padded_interest_items.append(intere_items)
                padded_interest_items_cate.append(intere_items_cate)
                padded_next_interest_items.append(next_intere_items)
                padded_next_interest_items_cate.append(next_intere_items_cate)

        users = torch.stack(padded_users, dim = 0).to(self.device)
        recent_items = torch.stack(padded_recent_items, dim = 0).to(self.device)
        recent_items_cate = torch.stack(padded_recent_items_cate, dim = 0).to(self.device)
        interest_items = torch.stack(padded_interest_items, dim = 0).to(self.device)
        interest_items_cate = torch.stack(padded_interest_items_cate, dim = 0).to(self.device)
        actions = torch.stack(actions_input, dim = 0).to(self.device)
        next_interest_items = torch.stack(padded_next_interest_items, dim = 0).to(self.device)
        next_interest_items_cate = torch.stack(padded_next_interest_items_cate, dim = 0).to(self.device)
        rewards = torch.tensor(rewards_input, dtype = torch.float).view(-1, 1).to(self.device)

        return users, recent_items, recent_items_cate, interest_items, interest_items_cate, actions, next_interest_items, next_interest_items_cate, rewards

    def update_users_ppo(self):
        # self.user_transition_dict = {'users': [], 'interest_items': [], 'interest_items_cate': [], 'recent_items': [], 'recent_items_cate': [], 'actions': [], 'next_interest_items': [], 'next_interest_items_cate': [], 'rewards': []}

        user_index = {}
        for idx, user_data in enumerate(self.user_transition_dict['users']):
            user_id = user_data[0].cpu().item()
            if user_id not in user_index.keys():
                user_index[user_id] = [idx]
            else:
                user_index[user_id].append(idx)
        indices = [user_index[u] for u in user_index.keys()]

        training_dataset = []
        for index in indices:
            data = {'users': [], 'interest_items': [], 'interest_items_cate': [], 'recent_items': [], 'recent_items_cate': [], 'actions': [], 'next_interest_items': [], 'next_interest_items_cate': [], 'rewards': []}
            for idx in index:
                data['users'].append(self.user_transition_dict['users'][idx])
                data['interest_items'].append(self.user_transition_dict['interest_items'][idx])
                data['interest_items_cate'].append(self.user_transition_dict['interest_items_cate'][idx])
                data['recent_items'].append(self.user_transition_dict['recent_items'][idx])
                data['recent_items_cate'].append(self.user_transition_dict['recent_items_cate'][idx])
                data['actions'].append(self.user_transition_dict['actions'][idx])
                data['next_interest_items'].append(self.user_transition_dict['next_interest_items'][idx])
                data['next_interest_items_cate'].append(self.user_transition_dict['next_interest_items_cate'][idx])
                data['rewards'].append(self.user_transition_dict['rewards'][idx])
            training_dataset.append(data)

        for training_data in training_dataset:
            users, recent_items, recent_items_cate, interest_items, interest_items_cate, actions, next_interest_items, next_interest_items_cate, rewards = \
            self.get_padded_data(training_data)

            td_target = rewards + self.gamma * self.critic_user(users, recent_items, recent_items_cate, next_interest_items, next_interest_items_cate)
            # print(f'rewards:{rewards.shape}')
            # print(f'self.critic_user:{self.critic_user(users, recent_items, recent_items_cate, next_interest_items, next_interest_items_cate).shape}')
            # print(f'td_target:{td_target.shape}')
            # exit()
            td_delta = td_target - self.critic_user(users, recent_items, recent_items_cate, interest_items, interest_items_cate)
            # print(f'real reward:{rewards}')
            # exit()
            advantage = self.compute_advantage(td_delta.cpu()).to(self.device)
            old_log_probs_all = torch.log(self.actor_user(users, recent_items, recent_items_cate, interest_items, interest_items_cate).gather(1, actions)).squeeze(-1).detach()
            old_log_probs = torch.sum(old_log_probs_all, dim=1)

            for _ in range(self.ppo_epochs_user):
                log_probs_all = torch.log(self.actor_user(users, recent_items, recent_items_cate, interest_items, interest_items_cate).gather(1, actions)).squeeze(-1)
                log_probs = torch.sum(log_probs_all, dim=1)
                # print(f'old_log_probs:{old_log_probs}')
                # print(f'log_probs:{log_probs}')
                # print(f'log_probs:{log_probs.shape}')
                ratio = torch.exp(log_probs - old_log_probs)
                # print(f'ratio:{ratio.shape}')
                surr1 = ratio * advantage
                # print(f'surr1:{surr1.shape}')
                surr2 = torch.clamp(ratio, 1 - self.eps,
                                    1 + self.eps) * advantage  # 截断
                # print(f'surr2:{surr2.shape}')
                actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
                # print(f'advantage:{advantage}')
                # print(f'ratio:{ratio}')
                # print(f'surr2:{surr2}')
                print(f'actor_loss:{actor_loss}')
                # print(f'self.critic_user:{self.critic_user(users, recent_items, recent_items_cate, interest_items, interest_items_cate).shape}')
                critic_loss = torch.mean(
                    F.mse_loss(self.critic_user(users, recent_items, recent_items_cate, interest_items, interest_items_cate), td_target.detach()))
                # print(f'td_target:{td_target}')
                print(f'critic_loss:{critic_loss}')
                self.actor_user_optimizer.zero_grad()
                self.critic_user_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                self.actor_user_optimizer.step()
                self.critic_user_optimizer.step()
            print('\n')
    # def update_users_ppo(self):
    #     #self.user_transition_dict = {'users': [], 'interest_items': [], 'recent_items': [], 'actions': [], 'next_interest_items': [], 'rewards': []}
    #     users = torch.stack(self.user_transition_dict['users'], dim = 0).to(self.device)
    #     interest_items = torch.stack(self.user_transition_dict['interest_items'], dim = 0).to(self.device)
    #     max_len = max([tensor.size(0) for tensor in self.user_transition_dict['recent_items']])
    #     padded_recent_items = []
    #     for tensor in self.user_transition_dict['recent_items']:
    #         if tensor.size(0) < max_len:
    #             pad_len = max_len - tensor.size(0)
    #             # 填充 tensor
    #             padded_tensor = F.pad(tensor, (0, pad_len))  # padding (left, right)
    #             padded_recent_items.append(padded_tensor)
    #         else:
    #             padded_recent_items.append(tensor)
    #     recent_items = torch.stack(padded_recent_items, dim = 0).to(self.device)
    #     actions = torch.stack(self.user_transition_dict['actions'], dim = 0).to(self.device)
    #     next_interest_items = torch.stack(self.user_transition_dict['next_interest_items'], dim = 0).to(self.device)
    #     rewards = torch.tensor(self.user_transition_dict['rewards'], dtype = torch.float).view(-1, 1).to(self.device)
    #     # print(f'users:{users.shape}')
    #     # print(f'interest_items:{interest_items.shape}')
    #     # print(f'recent_items:{recent_items.shape}')
    #     # print(f'actions:{actions.shape}')
    #     # print(f'next_interest_items:{next_interest_items.shape}')
    #     # print(f'rewards:{rewards.shape}')
    #     # exit()
    #     # states = torch.stack(transition_dict['states'], dim = 0).squeeze(1).float().to(self.device) # [batch_size, provider_len, categories_len+1]
    #     # actions = torch.stack(transition_dict['actions'], dim = 0).unsqueeze(-1).to(self.device) # [batch_size, provider_len, 1]
    #     # rewards = torch.tensor(transition_dict['rewards'], dtype = torch.float).view(-1, 1).to(self.device) # [batch_size, 1]
    #     # next_states = torch.stack(transition_dict['next_states'], dim = 0).squeeze(1).float().to(self.device) # [batch_size, provider_len, categories_len+1]
    #     # dones = torch.tensor(transition_dict['dones'],
    #     #                      dtype=torch.float).view(-1, 1).to(self.device)
    #     td_target = rewards + self.gamma * self.critic_user(users, recent_items, next_interest_items)

    #     td_delta = td_target - self.critic_user(users, recent_items, interest_items)
    #     # exit()
    #     advantage = self.compute_advantage(td_delta.cpu()).to(self.device)
    #     old_log_probs_all = torch.log(self.actor_user(users, recent_items, interest_items).gather(1, actions)).squeeze(-1).detach()
    #     old_log_probs = torch.sum(old_log_probs_all, dim=1)
    #     # print(f'old_log_probs_all:{old_log_probs_all.shape}')
    #     # print(f'old_log_probs_all:{old_log_probs_all}')
    #     # print(f'old_log_probs:{old_log_probs.shape}')
    #     # print(f'old_log_probs:{old_log_probs}')

    #     # print(f'td_target:{td_target.shape}')
    #     # print(f'td_delta:{td_delta.shape}')
    #     # print(f'advantage:{advantage.shape}')
    #     # print(f'old_log_probs:{old_log_probs.shape}')
    #     for _ in range(self.ppo_epochs):
    #         log_probs_all = torch.log(self.actor_user(users, recent_items, interest_items).gather(1, actions)).squeeze(-1)
    #         log_probs = torch.sum(log_probs_all, dim=1)
    #         # print(f'old_log_probs:{old_log_probs}')
    #         # print(f'log_probs:{log_probs}')
    #         # print(f'log_probs:{log_probs.shape}')
    #         ratio = torch.exp(log_probs - old_log_probs)
    #         # print(f'ratio:{ratio.shape}')
    #         surr1 = ratio * advantage
    #         # print(f'surr1:{surr1.shape}')
    #         surr2 = torch.clamp(ratio, 1 - self.eps,
    #                             1 + self.eps) * advantage  # 截断
    #         # print(f'surr2:{surr2.shape}')
    #         actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
    #         # print(f'advantage:{advantage}')
    #         # print(f'ratio:{ratio}')
    #         # print(f'surr2:{surr2}')
    #         # print(f'actor_loss:{actor_loss}')
    #         critic_loss = torch.mean(
    #             F.mse_loss(self.critic_user(users, recent_items, interest_items), td_target.detach()))
    #         # print(f'critic_loss:{critic_loss}')
    #         self.actor_user_optimizer.zero_grad()
    #         self.critic_user_optimizer.zero_grad()
    #         actor_loss.backward()
    #         critic_loss.backward()
    #         self.actor_user_optimizer.step()
    #         self.critic_user_optimizer.step()


    def save(self, save_dir_name):
        """Save the simulator status of current round"""
        utils.ensure_dir(save_dir_name)
        ID = utils.generate_id(self.config["simulator_dir"])
        file_name = f"{ID}-Round[{self.round_cnt}]-AgentNum[{self.config['agent_num']}]-{datetime.now().strftime('%Y-%m-%d-%H_%M_%S')}"
        self.file_name_path.append(file_name)
        save_file_name = os.path.join(save_dir_name, file_name + ".pkl")
        with open(save_file_name, "wb") as f:
            dill.dump(self.__dict__, f)
        self.logger.info("Current simulator Save in: \n" + str(save_file_name) + "\n")
        self.logger.info(
            "Simulator File Path (root -> node): \n" + str(self.file_name_path) + "\n"
        )
        utils.ensure_dir(self.config["ckpt_path"])
        cpkt_path = os.path.join(self.config["ckpt_path"], file_name + ".pth")
        self.recsys.save_model(cpkt_path)
        self.logger.info(
            "Current Recommender Model Save in: \n" + str(cpkt_path) + "\n"
        )

    @classmethod
    def restore(cls, restore_file_name, config, logger):
        """Restore the simulator status from the specific file"""
        with open(restore_file_name + ".pkl", "rb") as f:
            obj = cls.__new__(cls)
            obj.__dict__ = dill.load(f)
            obj.config, obj.logger = config, logger
            return obj

    def relevance_score_fn(self, score: float) -> float:
        """Return a similarity score on a scale [0, 1]."""
        # This will differ depending on a few things:
        # - the distance / similarity metric used by the VectorStore
        # - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
        # This function converts the euclidean norm of normalized embeddings
        # (0 is most similar, sqrt(2) most dissimilar)
        # to a similarity function (0 to 1)
        return 1.0 - score / math.sqrt(2)

    def create_new_memory_retriever(self):
        """Create a new vector store retriever unique to the agent."""
        # Define your embedding model
        embedding_size, embeddings_model = self.embedding_size, self.embedding_model
        # Initialize the vectorstore as empty
        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(
            embeddings_model.embed_query,
            index,
            InMemoryDocstore({}),
            {},
            relevance_score_fn=self.relevance_score_fn,
        )

        # If choose RecAgentMemory, you must use RecAgentRetriever rather than TimeWeightedVectorStoreRetriever.
        RetrieverClass = (
            RecAgentRetriever
            # if self.config["recagent_memory"] == "recagent"
            # else TimeWeightedVectorStoreRetriever
        )

        return RetrieverClass(
            vectorstore=vectorstore, embedding_function=embeddings_model.embed_query, other_score_keys=["importance"], now=self.now, k=5
        )

    def pause(self):
        self.play_event.clear()

    def play(self):
        self.play_event.set()


    def global_message(self, message: str):
        for i, agent in self.agents.items():
            agent.memory.add_memory(message, self.now)

    # def one_step_for_provider(self, agent_id, signal):
    #     """Run one step of an agent."""
    #     # self.play_event.wait()
    #     proagent = self.provider_agents[agent_id]
    #     # proagent.initialize_tok_mod(self.tokenizer, self.model)
    #     name = proagent.name
    #     # if self.round_cnt == 1:
    #     #     proagent.initialize_provider_profile()
    #     # proagent.update_round(self.round_cnt) #之前已经update过了
    #     # if not self.check_proactive(agent_id):
    #     #     return message
    #     user_interest_dict = self.data.get_user_interest_dict()
    #     # categories = self.data.get_item_categories()
    #     choose_genre = proagent.analyzing(categories=self.categories,
    #                                                         user_interest_dict=user_interest_dict,
    #                                                         round_cnt=self.round_cnt, signal = signal)
    #     # self.logger.info(f'Provider {name} has analyzed for the user feedback, and have a conclusion {analyze_conclusion}')
    #     # analyze_history = [analyze_prompt, analyze_result]  #None
    #     # action = proagent.extract_analyze_result(choose_genre)

    #     if len(proagent.new_round_item) > 0:
    #         new_item_id = proagent.new_round_item[-1]
    #         new_item_click = proagent.item_acc_click[new_item_id]
    #         new_item_genre = proagent.items[new_item_id]['genre']
    #         if choose_genre == new_item_genre:
    #             exploit = True
    #         else:
    #             exploit = False
    #         utils.save_action_records(self.round_cnt, name, new_item_click, exploit, self.config)

    #     retries = 0
    #     while retries < 5:
    #         try:
    #             generate_result = proagent.generating(choose_genre)
    #             item_name, item_genre, item_tags, item_description = utils.response_to_item(generate_result, choose_genre)
    #             self.upload_item(item_name, item_genre, item_tags, item_description, name)
    #             break
    #         except AttributeError as e:
    #             print(f"Error occurred: {e}, retrying...")
    #             if retries == 4:
    #                 raise ValueError('No valid item')
    #             retries += 1

    #     return message


    def one_step_for_provider(self, agent_id, signals):
        """Run one step of an agent."""
        # self.play_event.wait()
        if signals is not None:
            signal = signals[agent_id-1]
            # if signal == 0:
            #     signal = 10
        else:
            signal = None

        proagent = self.provider_agents[agent_id]
        # proagent.initialize_tok_mod(self.tokenizer, self.model)
        name = proagent.name
        # if self.round_cnt == 1:
        #     proagent.initialize_provider_profile()
        proagent.update_round(self.round_cnt)
        # if not self.check_proactive(agent_id):
        #     return message
        user_interest_dict = self.data.get_user_interest_dict()
        # categories = self.data.get_item_categories()
        decision_policy = self.config['provider_decision_policy']
        # if len(proagent.new_round_item) > 0:
        #     new_item_click =  proagent.item_acc_click[proagent.new_round_item[-1] ]
        #     utils.save_new_item_click_records(self.round_cnt, proagent.name, new_item_click, self.config)
        trust = True
        if self.config['change_trust']:
            threshold = random.uniform(0, 1)
            if threshold > proagent.trust:
                trust = False

        if (signal is not None) and (signal!=0) and trust:
            choose_genre = self.categories[signal-1]
            choice = 'Create'
            action = f'{proagent.name} choose to create video of genre {choose_genre}'
            proagent.belief = True
            proagent.latest_generate_genre = signal
            analyze_history = None # ''

        elif decision_policy not in ['random', 'LBR', 'CFD', 'SimuLine']:
            analyze_prompt, analyze_result = proagent.analyzing(categories=self.categories,
                                                                user_interest_dict=user_interest_dict,
                                                                round_cnt=self.round_cnt, signal = signal,
                                                                popular=self.genre_popular,
                                                                click=self.genre_click)
            # self.logger.info(f'Provider {name} has analyzed for the user feedback, and have a conclusion {analyze_conclusion}')
            if self.round_cnt <= 20 or self.config['provider_decision_policy'] == 'consistent':
                analyze_history = None # ''
            else:
                # analyze_history = [analyze_prompt, analyze_result]  #None
                analyze_history = None  #None
            choose_genre, choice, action = proagent.extract_analyze_result(self.categories, analyze_result)

            # if (signal is None) or (signal == 0) or (not trust):
            #     analyze_prompt, analyze_result = proagent.analyzing(categories=self.categories,
            #                                                         user_interest_dict=user_interest_dict,
            #                                                         round_cnt=self.round_cnt, signal = signal)
            #     # self.logger.info(f'Provider {name} has analyzed for the user feedback, and have a conclusion {analyze_conclusion}')
            #     if self.config['provider_decision_policy'] == 'consistent':
            #         analyze_history = ''
            #     else:
            #         analyze_history = [analyze_prompt, analyze_result]  #None
            #     choose_genre, choice, action = proagent.extract_analyze_result(self.categories, analyze_result)
            # else:
            #     # analyze_prompt = proagent.get_analyze_prompt(categories=self.categories,
            #     #                                                     user_interest_dict=user_interest_dict,
            #     #                                                     round_cnt=self.round_cnt)
            #     # analyze_result = f'[EXPLOIT]:: {self.categories[signal-1]}'
            #     # analyze_history = [analyze_prompt, analyze_result]
            #     # choose_genre, choice, action = proagent.extract_analyze_result(self.categories, analyze_result)
            #     choose_genre = self.categories[signal-1]
            #     choice = 'Create'
            #     action = f'{proagent.name} choose to create video of genre {choose_genre}'
            #     proagent.belief = True
            #     proagent.latest_generate_genre = signal
            #     analyze_history = ''
        else:
            analyze_result = ''
            analyze_prompt = ''
            analyze_history = None
            if decision_policy == 'random':
                choose_genre = random.choice(self.categories)
                choice = '[RAMDOM]'
                action = f'{proagent.name} choose to create video of genre {choose_genre}'
            elif decision_policy == 'LBR':
                choice = '[LBR]'
                item_num = len(proagent.new_round_item)
                creation_state = proagent.get_creation_state()
                if item_num != 0:
                    direction = np.random.randn(len(self.categories))  # 正太分布
                    direction = utils.L2norm(direction)
                    middle_state = creation_state + self.config['provider_lr'] * direction
                    user_vector = proagent.get_creation_utility_vector()
                    # print(f'user vector:{user_vector}')
                    middle_utility = np.dot(middle_state, user_vector)
                    ori_utility = np.dot(creation_state, user_vector)
                    # print(f'middle_utility:{middle_utility}')
                    # print(f'ori utility:{ori_utility}')
                    if middle_utility > ori_utility:
                        proagent.creation_state = utils.L2norm(middle_state)
                # print(f'{name} creation state:{proagent.creation_state}')
                max_index = np.argmax(proagent.creation_state)

                choose_genre = self.categories[max_index]
                action = f'{proagent.name} choose to create video of genre {choose_genre}'
                proagent.latest_generate_genre = max_index + 1
            elif decision_policy == 'CFD':
                choice = '[CFD]'
                item_num = len(proagent.new_round_item)
                creation_state = proagent.get_creation_state()
                if item_num != 0:
                    user_vector = proagent.get_creation_utility_vector()
                    print(f'user vector:{user_vector}')
                    proagent.creation_state = creation_state + self.config['provider_lr'] * user_vector
                    proagent.creation_state = utils.L2norm(proagent.creation_state)
                # print(f'{name} creation state:{proagent.creation_state}')
                max_index = np.argmax(proagent.creation_state)
                choose_genre = self.categories[max_index]
                action = f'{proagent.name} choose to create video of genre {choose_genre}'
                proagent.latest_generate_genre = max_index + 1
            elif decision_policy == 'SimuLine':
                choice = '[SimuLine]'
                # item_num = len(proagent.new_round_item)
                user_vector = proagent.get_creation_utility_vector()
                creation_prob_vec = []
                concentration_number = proagent.concerntration_num
                for i in range(len(user_vector)):
                    prob = - (1 - concentration_number) / (user_vector[i] + 1e-2)
                    creation_prob_vec.append(prob)
                creation_prob_vec = np.array(creation_prob_vec)
                # print(f'creation_prob:{creation_prob_vec}')
                softmax_vec = utils.softmax(creation_prob_vec)
                # print(f'soft_max_vec:{softmax_vec}')
                max_index = np.random.choice(len(softmax_vec), p=softmax_vec)
                choose_genre = self.categories[max_index]
                action = f'{proagent.name} choose to create video of genre {choose_genre}'
                proagent.latest_generate_genre = max_index + 1
            else:
                choose_genre = 'WRONG'
                choice = 'Wrong'
                raise ValueError(f'Not Valid decision_policy:{decision_policy}')
        if len(proagent.new_round_item) > 0:
            new_item_id = proagent.new_round_item[-1]
            new_item_click = proagent.item_acc_click[new_item_id]
            new_item_genre = proagent.items[new_item_id]['genre']
            if choose_genre == new_item_genre:
                exploit = True
            else:
                exploit = False
            utils.save_action_records(self.round_cnt, proagent.name, new_item_click, exploit, self.config)

        retries = 0
        while retries < 5:
            try:
                generate_result = proagent.generating(action, choice, choose_genre, analyze_history)
                item_name, item_genre, item_tags, item_description = utils.response_to_item(generate_result, choose_genre, self.config['data_name'])
                self.upload_item(item_name, item_genre, item_tags, item_description, name)
                break
            except AttributeError as e:
                print(f"Error occurred: {e}, retrying...")
                if retries == 4:
                    raise ValueError('No valid item')
                retries += 1

        return message


    def upload_item(self, name, genre, tags,  description, provider_name):
        provider_id = self.data.get_provider_id_by_name(provider_name)
        # if name == '':
        #     self.new_round_item.append((provider_id, -1))
        #     return -1
        try:
            max_item_id = max(self.data.get_all_item_ids())
        except:
            max_item_id = 0
        # print(f'all item id:{self.data.get_all_item_ids()}')
        # print(f'max_item_id:{max_item_id}')
        new_item_id = int(max_item_id) + 1   # 1 is the first item
        item_dict = {
            "name": name.strip(),
            "provider_name": provider_name,
            "provider_id": provider_id,
            "genre": genre,
            "upload_time": self.round_cnt,
            "tags": tags,
            "description": description.strip(),
            "inter_cnt": 0,
            "mention_cnt": 0,
        }
        self.data.items[new_item_id] = item_dict
        self.data.providers[provider_id]['items'].append(new_item_id)
        self.recsys.data = self.data
        self.data.item2provider[new_item_id] = provider_id
        self.provider_agents[provider_id].add_item(new_item_id, item_dict)
        self.provider_agents[provider_id].update_categories_times_and_skill(genre)
        self.new_round_item.append((provider_id, new_item_id))
        self.logger.info(f'<{provider_name}> create a new item:\n NAME:{name.strip()} GENRE:{genre} DESC:{description.strip()}')
        return new_item_id

    # def clear_items(self):
    #     self.data.items = {}

    def get_recent_items(self, item_recency):
        items = self.data.items
        recent_item_ids = []
        for item_id, item_dict in items.items():
            up_time = item_dict['upload_time']
            if self.round_cnt - up_time <= item_recency:
                recent_item_ids.append(item_id)
        return recent_item_ids

    def one_step_for_user_with_rec(self, agent_id, item_ids, rec_items):
        """Run one step of an agent."""
        # self.play_event.wait()
        # if not self.check_active(agent_id):
        #     return [
        #         Message(agent_id=agent_id, action="NO_ACTION", content="No action.")
        #     ]
        agent = self.agents[agent_id]
        name = agent.name
        # interest = agent.interest
        message = []
        # with lock:
        #     heapq.heappush(self.working_agents, agent)
        # if "REC" in choice:
        ids = []  # 被推荐的商品
        # self.rec_cnt[agent_id] += 1
        # self.logger.info(f"{name} enters the recommender system.")
        # self.round_msg.append(
        #     Message(
        #         agent_id=agent_id,
        #         action="RECOMMENDER",
        #         content=f"{name} enters the recommender system.",
        #     )
        # )
        # if self.round_cnt == 1:
        #     new_items = [i for p, i in self.new_round_item]
        #     # print(f'new items:{new_items}')
        #     item_ids, rec_items = self.recsys.get_random_items(agent_id, items=new_items)
        # else:
        # new_items = [i for p, i in self.new_round_item]
        # recent_items = self.get_recent_items()
        # item_ids, rec_items = self.recsys.get_full_sort_items(agent_id, item_set=recent_items)      # 要推荐的item。按顺序排列  itemids 是id， recitems 是item的描述
        # item_ids = item_ids[:self.config['TopK']]
        # rec_items = rec_items[:self.config['TopK']]
        # self.logger.info(f'rec items:{rec_items}')

        # duration = 2
        agent.reset_click()
        for true_rec_itemid, true_rec_item_description in zip(item_ids, rec_items): #
            # self.logger.info(
            #     f"{name} is recommended {true_rec_item_description}."
            # )
            # self.round_msg.append(
            #     Message(
            #         agent_id=agent_id,
            #         action="RECOMMENDER",
            #         content=f"{name} is recommended {true_rec_item_description}.",
            #     )
            # )
            observation = f"{name} is browsing the recommender system."

            observation = (
                observation
                + f" {name} is recommended {true_rec_item_description}."
            )
            # choice, action = agent.take_recommender_action(observation, self.now)
            choice, action = agent.take_click_action_with_rec(observation, self.now)
            # print(f'-----choice:{choice}------action:{action}')
            self.recsys.update_history_by_id(
                agent_id,
                [true_rec_itemid],
            )
            ids.extend([true_rec_itemid])

            if choice =='[WATCH]':
                # self.logger.info(f"{name} ({interest}) watches {true_rec_item_description}")
                # message.append(
                #     Message(
                #         agent_id=agent_id,
                #         action="RECOMMENDER",
                #         content=f"{name} watches {true_rec_item_description}.",
                #     )
                # )
                # self.round_msg.append(
                #     Message(
                #         agent_id=agent_id,
                #         action="RECOMMENDER",
                #         content=f"{name} watches {true_rec_item_description}.",
                #     )
                # )
                agent.update_watched_history(true_rec_item_description)
                agent.update_click()
                # if not self.config['signal_user']:
                #     self.recsys.update_positive_by_id(agent_id, true_rec_itemid)
                #     self.recsys.add_round_record(agent_id, true_rec_itemid, 1, self.round_cnt)
                #     self.recsys.add_train_data(agent_id, true_rec_itemid, 1)
                # else:
                #     self.interest_items[agent_id].insert(0, true_rec_itemid)
                #     self.interest_items[agent_id].pop()
                #     self.temp_user_transition_dict[agent_id]['rewards'] += 1.0
                #     self.record[agent_id].append(true_rec_itemid)
                self.recsys.update_positive_by_id(agent_id, true_rec_itemid)
                self.recsys.add_round_record(agent_id, true_rec_itemid, 1, self.round_cnt)
                self.genre_click[self.data.items[true_rec_itemid]['genre']] += 1
                if self.config['signal_user']:
                    self.recsys.add_train_data_for_signal(agent_id, true_rec_itemid, 1)
                else:
                    self.recsys.add_train_data(agent_id, true_rec_itemid, 1)

                self.update_click_to_providers(true_rec_itemid)  # 点击更新给供应商
                self.update_exposure_to_providers(true_rec_itemid)


                # item_descriptions = self.data.get_item_description_by_name([true_rec_item_description])

                # observation = f"{name} has just finished watching {true_rec_item_description};;{item_descriptions[0]}."
                # feelings = agent.generate_feeling(
                #     observation, self.now + timedelta(hours=duration)
                # )
                # provider_id = self.data.get_provider_id_by_item_id(true_rec_itemid)
                # self.provider_agents[provider_id].upload_comments(feelings)
                # self.logger.info(f"{name}({interest}) feels: {feelings}")
                #
                # self.round_msg.append(
                #     Message(
                #         agent_id=agent_id,
                #         action="RECOMMENDER",
                #         content=f"{name} feels: {feelings}",
                #     )
                # )
            elif choice == '[SKIP]':
                # self.logger.info(f"{name} ({interest}) skip the video.")
                # if not self.config['signal_user']:
                if self.config['signal_user']:
                    self.recsys.add_train_data_for_signal(agent_id, true_rec_itemid, 0)
                else:
                    self.recsys.add_train_data(
                        agent_id, true_rec_itemid, 0
                    )
                self.recsys.add_round_record(agent_id, true_rec_itemid, 0, self.round_cnt)
                self.update_exposure_to_providers(true_rec_itemid)
                # self.round_msg.append(
                #     Message(
                #         agent_id=agent_id,
                #         action="RECOMMENDER",
                #         content=f"{name} looks next page.",
                #     )
                # )
            else:
                self.logger.info(f"{name} leaves the recommender system.")
                # self.round_msg.append(
                #     Message(
                #         agent_id=agent_id,
                #         action="RECOMMENDER",
                #         content=f"{name} leaves the recommender system.",
                #     )
                # )
                break
        # self.recsys.round_record[agent_id].append(ids)

        return message

    def one_step_for_user(self, agent_id):
        """Run one step of an agent."""
        self.play_event.wait()
        # if not self.check_active(agent_id):
        #     return [
        #         Message(agent_id=agent_id, action="NO_ACTION", content="No action.")
        #     ]
        agent = self.agents[agent_id]
        name = agent.name
        interest = agent.interest
        message = []
        # choice, observation = agent.take_action(self.now)  # 选择进入RS，进入社交媒体，还是什么都不做
        # with lock:
        #     heapq.heappush(self.working_agents, agent)
        # if "REC" in choice:
        ids = []  # 被推荐的商品
        # self.rec_cnt[agent_id] += 1
        self.logger.info(f"{name} enters the recommender system.")
        self.round_msg.append(
            Message(
                agent_id=agent_id,
                action="RECOMMENDER",
                content=f"{name} enters the recommender system.",
            )
        )
        # item_ids, rec_items = self.recsys.get_full_sort_items(agent_id)    # 要推荐的item。按顺序排列  itemids 是id， recitems 是item的描述
        new_items = [i for p, i in self.new_round_item]
        # print(f'new items:{new_items}')
        item_ids, rec_items = self.recsys.get_random_items(agent_id, items=new_items)
        duration = 2
        for true_rec_itemid, true_rec_item_description in zip(item_ids, rec_items):
        # while not leave:
            self.logger.info(
                f"{name}({interest}) is recommended {true_rec_item_description}."
            )
            self.round_msg.append(
                Message(
                    agent_id=agent_id,
                    action="RECOMMENDER",
                    content=f"{name} is recommended {true_rec_item_description}.",
                )
            )
            observation = f"{name} is browsing the recommender system."

            observation = (
                observation
                + f" {name} is recommended {true_rec_item_description}."
            )
            # choice, action = agent.take_recommender_action(observation, self.now)
            choice, action = agent.take_click_action(observation, self.now)
            # print(f'-----choice:{choice}------action:{action}')
            self.recsys.update_history_by_id(
                agent_id,
                [true_rec_itemid],
            )
            ids.extend([true_rec_itemid])

            if choice =='[WATCH]':

                self.logger.info(f"{name}({interest}) watches {true_rec_item_description}")
                message.append(
                    Message(
                        agent_id=agent_id,
                        action="RECOMMENDER",
                        content=f"{name} watches {true_rec_item_description}.",
                    )
                )
                self.round_msg.append(
                    Message(
                        agent_id=agent_id,
                        action="RECOMMENDER",
                        content=f"{name} watches {true_rec_item_description}.",
                    )
                )
                agent.update_watched_history(true_rec_item_description)
                self.recsys.update_positive_by_id(agent_id, true_rec_itemid)

                # for i in range(self.recsys.page_size):
                #     # print(f'action:{action}')
                #     # print(f'666{item_ids[page * self.recsys.page_size + i]}')
                #     try:
                #         exposed_item_id = item_ids[page * self.recsys.page_size + i]
                #     except IndexError:
                #         continue
                # if i == action - 1:
                # update
                # print(f'yes : i inter')
                self.recsys.add_train_data(agent_id, true_rec_itemid, 1)
                # self.recsys.add_round_provider_click_data()
                # print(f'update click to provider:{true_rec_itemid}')
                self.update_click_to_providers(true_rec_itemid)  # 点击更新给供应商
                self.update_exposure_to_providers(true_rec_itemid)
                # else:
                #     self.recsys.add_train_data(agent_id, exposed_item_id, 0)
                #     self.update_exposure_to_providers(exposed_item_id) # 把曝光更新给每一个供应商


                item_descriptions = self.data.get_item_description_by_name([true_rec_item_description])

                observation = f"{name} has just finished watching {true_rec_item_description};;{item_descriptions[0]}."
                feelings = agent.generate_feeling(
                    observation, self.now + timedelta(hours=duration)
                )
                provider_id = self.data.get_provider_id_by_item_id(true_rec_itemid)
                self.provider_agents[provider_id].upload_comments(feelings)
                self.logger.info(f"{name}({interest}) feels: {feelings}")

                self.round_msg.append(
                    Message(
                        agent_id=agent_id,
                        action="RECOMMENDER",
                        content=f"{name} feels: {feelings}",
                    )
                )

            else:
                self.logger.info(f"{name}({interest}) skip the video.")
                self.recsys.add_train_data(
                    agent_id, true_rec_itemid, 0
                )
                self.update_exposure_to_providers(true_rec_itemid)
                self.round_msg.append(
                    Message(
                        agent_id=agent_id,
                        action="RECOMMENDER",
                        content=f"{name} looks next page.",
                    )
                )

        self.logger.info(f"{name} leaves the recommender system.")
        self.round_msg.append(
            Message(
                agent_id=agent_id,
                action="RECOMMENDER",
                content=f"{name} leaves the recommender system.",
            )
        )

        # self.recsys.round_record[agent_id].append(ids)

        return message


    def update_exposure_to_providers(self, exposed_item_id):
        # print(f'item2provider:{self.item2provider}')
        belong_provider_agent = self.provider_agents[self.data.item2provider[exposed_item_id]]
        # print(f'update exposure to {belong_provider_agent.name}')
        belong_provider_agent.update_exposure(exposed_item_id, self.round_cnt)

    def update_click_to_providers(self, clicked_item_id):

        belong_provider_agent = self.provider_agents[self.data.item2provider[clicked_item_id]]
        # print(f'update click to {belong_provider_agent.name}')
        belong_provider_agent.update_click(clicked_item_id, self.round_cnt)


    def construct_generate_prompts(self, analyze_responses_list, analyze_prompt_list):
        data_list = []
        choose_genre = ''
        for i, pro_id in tqdm(enumerate(self.active_proagents)): #tqdm(range(1, self.config["provider_agent_num"]+1)):
            proagent = self.provider_agents[pro_id]
            name = proagent.name
            profile_text = proagent.get_profile_text()

            # categories = utils.get_item_categories()
            decision_policy = self.config['provider_decision_policy']
            if decision_policy not in ['random', 'LBR', 'CFD']:
                analyze_result = analyze_responses_list[i]
                analyze_prompt = analyze_prompt_list[i]
                choose_genre, choice, action = proagent.extract_analyze_result(self.categories, analyze_result)
            else:
                analyze_result = ''
                analyze_prompt = ''
                if decision_policy == 'random':
                    choose_genre = random.choice(self.categories)
                    choice = '[RAMDOM]'
                    action = f'{proagent.name} choose to create video of genre {choose_genre}'
                elif decision_policy == 'LBR':
                    choice = '[LBR]'
                    item_num = len(proagent.new_round_item)
                    creation_state = proagent.get_creation_state()
                    if item_num != 0:

                        direction = utils.L2norm(direction)
                        middle_state = creation_state + self.config['provider_lr'] * direction
                        user_vector = proagent.get_creation_utility_vector()
                        print(f'user vector:{user_vector}')
                        middle_utility = np.dot(middle_state, user_vector)
                        ori_utility = np.dot(creation_state, user_vector)
                        print(f'middle_utility:{middle_utility}')
                        print(f'ori utility:{ori_utility}')
                        if middle_utility > ori_utility:
                            proagent.creation_state = utils.L2norm(middle_state)
                    print(f'{name} creation state:{proagent.creation_state}')
                    max_index = np.argmax(proagent.creation_state)

                    choose_genre = self.categories[max_index]
                    action = f'{proagent.name} choose to create video of genre {choose_genre}'
                elif decision_policy == 'CFD':
                    choice = '[CFD]'
                    item_num = len(proagent.new_round_item)
                    creation_state = proagent.get_creation_state()
                    if item_num != 0:
                        user_vector = proagent.get_creation_utility_vector()
                        print(f'user vector:{user_vector}')
                        proagent.creation_state = creation_state + self.config['provider_lr'] * user_vector
                        proagent.creation_state = utils.L2norm(proagent.creation_state)
                    print(f'{name} creation state:{proagent.creation_state}')
                    max_index = np.argmax(proagent.creation_state)
                    choose_genre = self.categories[max_index]
                    action = f'{proagent.name} choose to create video of genre {choose_genre}'


                else:
                    choose_genre = 'WRONG'
                    choice =  'Wrong'
                    raise ValueError(f'Not Valid decision_policy:{decision_policy}')

            # print(choice)
            # print(choose_genre)
            # print(action)
            recent_creation = proagent.get_recent_creation(category=choose_genre)
            single_dict = dict({"system": profile_text,
                                "instruction":  (
                                                f"Based on the analysis, {action}"
                                                f"\n Please create a brand new item in the {choose_genre} genre."
                                                "\n Return the results strictly according to the following JSON dictionary format: \n ```json"
                                                '''\n {"name": "item_name", "genre": "''' + choose_genre + '''", "tags": [tag1, tag2, tag3], "description": "item_description_text"}'''
                                ),
                                "input": f"You can draw inspiration from {name}'s previous creation on genre {choose_genre}, but cannot replicate them identically.\n{recent_creation}" if (recent_creation != None and choice != '[RANDOM]') else "",
                                         # f"{name} recently created {proagent.items[proagent.new_round_item[-1]]} on recommender system.",
                                "output": "",
                                "history": [
                                    [analyze_prompt, analyze_result],
                                ]}
                               )
            data_list.append(single_dict)

        # with open(f'round{self.round_cnt}.json', "w") as file:
        #     json.dump(data_list, file)

        return data_list, choose_genre

    def provider_round(self, signals):
        if self.config["execution_mode"] == "parallel":
            # batch_size = 200
            futures = []
            with tqdm(total=len(self.active_proagents), desc='Provider Processing...') as pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
                    for i in self.active_proagents:
                        # futures.append(executor.submit(self.one_step_for_provider, i, signals[i-1]))
                        futures.append(executor.submit(self.one_step_for_provider, i, signals))
                    # for b in range(1, self.config["provider_agent_num"] + 1, batch_size):
                    #     futures = [executor.submit(self.one_step_for_provider, i) for i in
                    #                range(b, min(self.config["provider_agent_num"] + 1, b + batch_size))]
                        # 等待当前批次完成
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        pbar.update(1)
                            # print(f'mission complete:{result}')
        else:
            for i in tqdm(self.active_proagents):
                self.one_step_for_provider(i)

        # if not self.config['signal']:
        #     return None

        genre_list_index = [0]*self.config['provider_agent_num']
        for i in self.active_proagents:
            genre_list_index[i-1] = self.provider_agents[i].latest_generate_genre
        state_one_hot = utils.int_to_onehot(genre_list_index, self.action_dim, self.device)
        new_state = state_one_hot.unsqueeze(0)
        return genre_list_index, new_state
        # self.logger.info(f'active provider number:{len(self.active_proagents)}')

    def update_round(self, state):
        self.round_cnt = self.round_cnt + 1
        self.new_round_item.clear()
        self.active_proagents.clear()
        self.active_agents.clear()
        self.recsys.round_record.clear()

        if self.config['rec_model'] not in ['Random']:
            if self.config['signal_user']:
                if self.round_cnt >= 10 and self.round_cnt % 5 == 0:
                    self.recsys.train_signal()
                    # self.recsys.clear_train_data()
            elif self.round_cnt % 5 == 1:
                if self.config['rec_model'] == 'BPR':
                    self.recsys.train_BPR()
                else:
                    self.recsys.train()
                    # if self.round_cnt % 25 == 0:
                    #     self.recsys.clear_train_data()
        if self.config['rec_save_checkpoint'] and self.round_cnt % 50 == 0:
            self.recsys.save_model(os.path.join(self.config['rec_checkpoint_save_path'], self.config['rec_model'] + f'_{self.round_cnt}_' + self.config['data_name'] + '.pth'))
        if self.config['ppo_save_checkpoint'] and self.round_cnt % 50 == 0:
            torch.save(self.actor.state_dict(), self.config['ppo_checkpoint_save_path'] + f'/actor_{self.round_cnt}_' + self.config['data_name'] + '.pth')
            torch.save(self.critic.state_dict(), self.config['ppo_checkpoint_save_path'] + f'/critic_{self.round_cnt}_' + self.config['data_name'] + '.pth')
        # if not self.config['signal_user'] and self.config['rec_model'] not in ['Random']:
        #     if self.round_cnt % 5 == 1:
        #         if self.config['rec_model'] == 'BPR':
        #             self.recsys.train_BPR()
        #         else:
        #             self.recsys.train()
        # elif self.config['signal_user']:
        #     if self.round_cnt >= 100 and self.round_cnt % 5 == 0:
        #         # print('\nUpdate users signal\n')
        #         self.update_users_ppo()
        #         # self.user_transition_dict = {'users': [], 'interest_items': [], 'interest_items_cate': [], 'recent_items': [], 'recent_items_cate': [], 'actions': [], 'next_interest_items': [], 'next_interest_items_cate': [], 'rewards': []}
        #     if self.config['save_checkpoint'] and self.round_cnt % 100 == 0:
        #         torch.save(self.actor_user.state_dict(), self.config['checkpoint_save_path'] + f'/actor_user_{self.round_cnt}.pth')
        #         torch.save(self.critic_user.state_dict(), self.config['checkpoint_save_path'] + f'/critic_user_{self.round_cnt}.pth')
        #         torch.save(self.actor.state_dict(), self.config['checkpoint_save_path'] + f'/actor_{self.round_cnt}.pth')
        #         torch.save(self.critic.state_dict(), self.config['checkpoint_save_path'] + f'/critic_{self.round_cnt}.pth')
        #     if self.round_cnt >= 600:
        #         self.user_transition_dict['users'] = self.user_transition_dict['users'][self.dict_count[0]:]
        #         self.user_transition_dict['interest_items'] = self.user_transition_dict['interest_items'][self.dict_count[0]:]
        #         self.user_transition_dict['interest_items_cate'] = self.user_transition_dict['interest_items_cate'][self.dict_count[0]:]
        #         self.user_transition_dict['recent_items'] = self.user_transition_dict['recent_items'][self.dict_count[0]:]
        #         self.user_transition_dict['recent_items_cate'] = self.user_transition_dict['recent_items_cate'][self.dict_count[0]:]
        #         self.user_transition_dict['actions'] = self.user_transition_dict['actions'][self.dict_count[0]:]
        #         self.user_transition_dict['next_interest_items'] = self.user_transition_dict['next_interest_items'][self.dict_count[0]:]
        #         self.user_transition_dict['next_interest_items_cate'] = self.user_transition_dict['next_interest_items_cate'][self.dict_count[0]:]
        #         self.user_transition_dict['rewards'] = self.user_transition_dict['rewards'][self.dict_count[0]:]
        #         self.dict_count.pop(0)
            # if self.round_cnt % 25 == 0:
            #     self.user_transition_dict = {'users': [], 'interest_items': [], 'interest_items_cate': [], 'recent_items': [], 'recent_items_cate': [], 'actions': [], 'next_interest_items': [], 'next_interest_items_cate': [], 'rewards': []}


        for i in tqdm(range(1, self.config["provider_agent_num"]+1)):
            proagent = self.provider_agents[i]
            name = proagent.name
            proagent.update_round(self.round_cnt)
                # self.active_proagents.append(i)
            if self.round_cnt == 1:
                if name in self.provider_profile_dict.keys():
                    profile_text = proagent.initialize_provider_profile(profile=self.provider_profile_dict[name])
                else:
                    profile_text = proagent.initialize_provider_profile(profile=None)
                    self.provider_profile_dict[name] = profile_text
                    with open(self.config['profile_path'], 'w') as f:
                        json.dump(self.provider_profile_dict, f, indent=4)
            if self.check_proactive(i):
                continue
        for i in tqdm(range(1, self.config["agent_num"] + 1)):
            if self.check_active(i):
                continue
        self.logger.info(f'active provider number:{len(self.active_proagents)}')
        self.logger.info(f'active user number:{len(self.active_agents)}')

        if not self.config['signal']:
            return None

        signals = self.sample_signals(state)
        return signals

    def sample_signals(self, state):
        action = self.take_action(state)
        signals = []
        for i in range(action.shape[0]):
            signals.append(action[i].item())
        return signals

    def construct_analyze_prompts(self):
        data_list = []
        for i in tqdm(self.active_proagents): #tqdm(range(1, self.config["provider_agent_num"]+1)):
            proagent = self.provider_agents[i]
            name = proagent.name
            profile_text = proagent.get_profile_text()

            # categories = self.data.get_item_categories()
            prompt = proagent.get_analyze_prompt(categories=self.categories,
                                                 round_cnt=self.round_cnt,
                                                 user_interest_dict=self.data.get_user_interest_dict())
            single_dict = dict({"system": profile_text,
                               "instruction": prompt,
                               "input": '',
                               "output": "",
                                "history": []
                                })
            data_list.append(single_dict)

        # with open(f'round{self.round_cnt}.json', "w") as file:
        #     json.dump(data_list, file)

        return data_list

    def get_genre_item_count(self):
        # cates = self.data.get_item_categories()
        count_dict = {k: 0 for k in self.categories}
        for pro_id, item_id in self.new_round_item:
            genre = self.data.items[item_id]['genre']
            count_dict[genre] += 1
        return count_dict


    def check_proactive(self, index: int):
        # If agent's previous action is completed, reset the event
        proagent = self.provider_agents[index]
        if (
            self.active_proagent_threshold
            and len(self.active_proagents) >= self.active_proagent_threshold
        ):
            return False

        if self.config['with_leave'] and index in self.leave_providers:
            return False

        active_prob = proagent.get_active_prob(self.active_method)
        random_state = np.random.random()
        # print(f'random state:{random_state} \t active prob :{active_prob}')
        if random_state > active_prob:
            proagent.no_action_round += 1  # 随机数大于 则 没action
            return False  # 不 active
        # 如果过去几轮创作的item，item一个点击都没有，则离开
        if self.round_cnt >= self.config['reranking_start_step'] + 10:
            if self.config['with_leave'] and len(proagent.new_round_item) >= 10:
                acc_click = 0
                for item_id in proagent.new_round_item[-10:]:
                    acc_click += proagent.item_acc_click[item_id]
                if acc_click == 0:
                    self.logger.info(f'Round {self.round_cnt}: provider <{proagent.name}> Leaves Forever...')
                    proagent.active_prob = 0.0
                    self.leave_providers.append(index)
                    self.logger.info(f'Round {self.round_cnt}: leave providers {self.leave_providers}')
                    return False
        self.active_proagents.append(index)
        return True

    def check_active(self, index: int):
        # If agent's previous action is completed, reset the event
        agent = self.agents[index]
        if (
            self.active_agent_threshold
            and len(self.active_agents) >= self.active_agent_threshold
        ):
            return False

        active_prob = agent.get_active_prob(self.active_method)
        if np.random.random() > active_prob:
            agent.no_action_round += 1 # 随机数大于 则 没action
            return False
        self.active_agents.append(index)
        return True

    def get_user_feedbacks(self):
        recent_items = self.get_recent_items(item_recency=self.config['item_recency'])
        if self.config['signal_user']:
            if self.config['rec_model'] == 'DIN':
                item_ids_dict, rec_items_dict = self.recsys.get_full_sort_items_for_signal(user_list=self.active_agents,
                                                                                    round_cnt=self.round_cnt,
                                                                                    item_set=recent_items)
            else:
                item_ids_dict, rec_items_dict = self.recsys.get_full_sort_items_for_Signal_MF(user_list=self.active_agents,
                                                                                    round_cnt=self.round_cnt,
                                                                                    item_set=recent_items)
        elif self.config['rec_model'] == "DIN":
            item_ids_dict, rec_items_dict = self.recsys.get_full_sort_items_for_users(user_list=self.active_agents,
                                                                                  round_cnt=self.round_cnt,
                                                                                  item_set=recent_items)               # 要推荐的item。按顺序排列  itemids 是id， recitems 是item的描述
        # elif self.config['rec_model'] == 'Signal':
        #     item_ids_dict, rec_items_dict = self.recsys.get_full_sort_items_for_signal(user_list=self.active_agents,
        #                                                                           round_cnt=self.round_cnt,
        #                                                                           item_set=recent_items)
        elif self.config['rec_model'] in  ["MF", 'Random', 'BPR', 'Pop']:
            item_ids_dict, rec_items_dict = self.recsys.get_full_sort_items_for_MF(user_list=self.active_agents,
                                                                                   round_cnt=self.round_cnt,
                                                                                  item_set=recent_items)


        if self.config["execution_mode"] == "parallel":
            # batch_size = 200
            futures = []
            with tqdm(total=len(self.active_agents), desc='User Processing...') as pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
                    for i in self.active_agents:
                        item_ids = item_ids_dict[i]  #[:self.config['TopK']]
                        rec_items = rec_items_dict[i]  #[:self.config['TopK']]
                        futures.append(executor.submit(self.one_step_for_user_with_rec,
                                                       i,
                                                       item_ids,
                                                       rec_items))

                    for future in concurrent.futures.as_completed(futures):
                        msgs = future.result()
                        pbar.update(1)
                # for b in range(1, self.config["agent_num"]+1, batch_size):
                #     futures = [executor.submit(self.one_step_for_user_with_rec, i) for i in range(b, min(self.config["agent_num"]+1, b+batch_size))]
                #     # 等待当前批次完成
                #     for future in concurrent.futures.as_completed(futures):
                #         result = future.result()
                #         # print(f'mission complete:{result}')
        else:
            for i in tqdm(self.active_agents, desc='User Processing...'):
                item_ids = item_ids_dict[i]  #[:self.config['TopK']]
                rec_items = rec_items_dict[i]  #[:self.config['TopK']]
                self.one_step_for_user_with_rec(i, item_ids, rec_items)

        # self.logger.info(f'active user number:{len(self.active_agents)}')
        self.recsys.save_round_interaction(self.round_cnt)
        # self.recsys.train()
        # self.save(os.path.join(self.config["simulator_dir"]))

        user_reward = []

        for i in self.active_agents:
            user_reward.append(self.agents[i].return_click())

        provider_rewards = []

        for provider_id in self.active_proagents:
            proagent = self.provider_agents[provider_id]
            last_new_item = proagent.new_round_item[-1]
            last_new_item_click = proagent.item_acc_click[last_new_item]
            provider_rewards.append(last_new_item_click)
            if self.config['change_trust']:
                if len(proagent.new_round_item) <=1:
                    proagent.update_last_reward(last_new_item_click)
                    continue
                proagent.update_trust(last_new_item_click)
                proagent.update_last_reward(last_new_item_click)

            # upload_time = self.data.items[last_new_item]['upload_time']
            # item_age = self.round_cnt - upload_time
            # if item_age >= self.config['item_recency']:
            #     active_round =  self.config['item_recency']
            # else:
            #     active_round = item_age
            # item_reward_per_round = last_new_item_click/active_round

        if len(provider_rewards) == 0:
            return np.sum(user_reward), 0
        else:
            return np.sum(user_reward), np.sum(provider_rewards)

    # def get_user_feedbacks_with_signal(self):
    #     recent_items = self.get_recent_items(item_recency=self.config['item_recency'])

    #     item_ids_dict, rec_items_dict = self.recsys.get_full_sort_items_for_signal(user_list=self.active_agents,
    #                                                                               round_cnt=self.round_cnt,
    #                                                                               item_set=recent_items) 
    #     if self.config["execution_mode"] == "parallel":
    #         # batch_size = 200
    #         futures = []
    #         with tqdm(total=len(self.active_agents), desc='User Processing...') as pbar:
    #             with concurrent.futures.ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
    #                 for i in self.active_agents:
    #                     item_ids = item_ids_dict[i]  #[:self.config['TopK']]
    #                     rec_items = rec_items_dict[i]  #[:self.config['TopK']]
    #                     futures.append(executor.submit(self.one_step_for_user_with_rec,
    #                                                    i,
    #                                                    item_ids,
    #                                                    rec_items))

    #                 for future in concurrent.futures.as_completed(futures):
    #                     msgs = future.result()
    #                     pbar.update(1)
    #             # for b in range(1, self.config["agent_num"]+1, batch_size):
    #             #     futures = [executor.submit(self.one_step_for_user_with_rec, i) for i in range(b, min(self.config["agent_num"]+1, b+batch_size))]
    #             #     # 等待当前批次完成
    #             #     for future in concurrent.futures.as_completed(futures):
    #             #         result = future.result()
    #             #         # print(f'mission complete:{result}')
    #     else:
    #         for i in tqdm(self.active_agents, desc='User Processing...'):
    #             item_ids = item_ids_dict[i]  #[:self.config['TopK']]
    #             rec_items = rec_items_dict[i]  #[:self.config['TopK']]
    #             self.one_step_for_user_with_rec(i, item_ids, rec_items)

    #     # self.recsys.save_round_interaction(self.round_cnt)

    #     user_reward = []
    #     self.dict_count.append(len(self.active_agents))
    #     for i in self.active_agents:
    #         user_reward.append(self.agents[i].return_click())
    
    #     provider_rewards = []

    #     for provider_id in self.active_proagents:
    #         proagent = self.provider_agents[provider_id]
    #         last_new_item = proagent.new_round_item[-1]
    #         last_new_item_click = proagent.item_acc_click[last_new_item]
    #         provider_rewards.append(last_new_item_click)
    #         if len(proagent.new_round_item) <=1:
    #             proagent.update_last_reward(last_new_item_click)
    #             continue
    #         proagent.update_trust(last_new_item_click)

    #         # upload_time = self.data.items[last_new_item]['upload_time']
    #         # item_age = self.round_cnt - upload_time
    #         # if item_age >= self.config['item_recency']:
    #         #     active_round =  self.config['item_recency']
    #         # else:
    #         #     active_round = item_age
    #         # item_reward_per_round = last_new_item_click/active_round

    #     if len(provider_rewards) == 0:
    #         return np.sum(user_reward), 0
    #     else:
    #         return np.sum(user_reward), np.sum(provider_rewards)

    # def get_user_feedbacks_with_signal(self):
    #     recent_items = self.get_recent_items(item_recency=self.config['item_recency'])
    #     recommend_list = []
    #     item_ids_dict = {}
    #     rec_items_dict = {}
    #     self.temp_user_transition_dict = {}
    #     for user in self.active_agents:
    #         print(f'user {user} interest items:{self.interest_items[user]}')
    #         self.temp_user_transition_dict[user] = {'user': None, 'interest_items': None, 'interest_items_cate': None, 'recent_items': None, 'recent_items_cate': None, 'actions': None, 'next_interest_items': None, 'next_interest_items_cate': None, 'rewards': 0}
    #         # items = [item for item in recent_items if item not in self.record[user]]
    #         user_list = [user for _ in recent_items]
    #         items = []
    #         for item in recent_items:
    #             if item not in self.record[user]:
    #                 items.append(item)
    #             else:
    #                 items.append(0)
    #         items_cate = [self.data.get_cate_id_by_item(i) for i in recent_items]
    #         interest_items_cate = [self.data.get_cate_id_by_item(i) for i in self.interest_items[user]]
    #         user_tensor = torch.tensor(user_list).unsqueeze(0).to(self.device)
    #         items_tensor = torch.tensor(items).unsqueeze(0).to(self.device)
    #         items_cate_tensor = torch.tensor(items_cate).unsqueeze(0).to(self.device)
    #         interest_items = torch.stack([torch.tensor(self.interest_items[user]) for _ in recent_items]).unsqueeze(0).to(self.device)
    #         interest_items_cate_tensor = torch.stack([torch.tensor(interest_items_cate) for _ in recent_items]).unsqueeze(0).to(self.device)
    #         # print(f'user_tensor:{user_tensor.shape}') # [18]
    #         # print(f'items_tensor:{items_tensor.shape}') # [18]
    #         # print(f'items_cate_tensor:{items_cate_tensor.shape}') # [18]
    #         # print(f'interest_items:{interest_items.shape}') # [18, 5]
    #         # print(f'interest_items_cate_tensor:{interest_items_cate_tensor.shape}') # [18, 5]
    #         # exit()
    #         action_pro = self.actor_user(user_tensor, items_tensor, items_cate_tensor, interest_items, interest_items_cate_tensor)
    #         print(f'action_pro:{action_pro}')
    #         # samples = torch.multinomial(action_pro, self.config['TopK'], replacement=False)
    #         _, samples = torch.topk(action_pro, self.config['TopK'], dim = 1)
    #         samples = samples.squeeze()
    #         recom_items = [items[index] for index in samples]
    #         recommend_list.append(recom_items)
    #         self.temp_user_transition_dict[user]['user'] = user_tensor.squeeze(0)
    #         self.temp_user_transition_dict[user]['interest_items'] = interest_items.squeeze(0)
    #         self.temp_user_transition_dict[user]['recent_items'] = items_tensor.squeeze(0)
    #         self.temp_user_transition_dict[user]['recent_items_cate'] = items_cate_tensor.squeeze(0)
    #         self.temp_user_transition_dict[user]['interest_items_cate'] = interest_items_cate_tensor.squeeze(0)
    #         self.temp_user_transition_dict[user]['actions'] = samples

    #     for user, user_recommend_list in zip(self.active_agents, recommend_list):
    #         item_names = self.data.get_item_names(user_recommend_list)
    #         item_tags = self.data.get_item_tags(user_recommend_list)
    #         description = self.data.get_item_description_by_id(user_recommend_list)
    #         items = [
    #             item_names[i]
    #             + ";; Genre: "
    #             + self.data.get_genres_by_ids([user_recommend_list[i]])[0]
    #             + ";; Tags: "
    #             + item_tags[i]
    #             + ";; Description: "
    #             + description[i]
    #             for i in range(len(item_names))
    #         ]
    #         item_ids_dict[user] = user_recommend_list  # for user index in users[index]
    #         rec_items_dict[user] = items

    #     if self.config["execution_mode"] == "parallel":
    #         # batch_size = 200
    #         futures = []
    #         with tqdm(total=len(self.active_agents), desc='User Processing...') as pbar:
    #             with concurrent.futures.ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
    #                 for i in self.active_agents:
    #                     item_ids = item_ids_dict[i]  #[:self.config['TopK']]
    #                     rec_items = rec_items_dict[i]  #[:self.config['TopK']]
    #                     futures.append(executor.submit(self.one_step_for_user_with_rec,
    #                                                    i,
    #                                                    item_ids,
    #                                                    rec_items))

    #                 for future in concurrent.futures.as_completed(futures):
    #                     msgs = future.result()
    #                     pbar.update(1)
    #             # for b in range(1, self.config["agent_num"]+1, batch_size):
    #             #     futures = [executor.submit(self.one_step_for_user_with_rec, i) for i in range(b, min(self.config["agent_num"]+1, b+batch_size))]
    #             #     # 等待当前批次完成
    #             #     for future in concurrent.futures.as_completed(futures):
    #             #         result = future.result()
    #             #         # print(f'mission complete:{result}')
    #     else:
    #         for i in tqdm(self.active_agents, desc='User Processing...'):
    #             item_ids = item_ids_dict[i]  #[:self.config['TopK']]
    #             rec_items = rec_items_dict[i]  #[:self.config['TopK']]
    #             self.one_step_for_user_with_rec(i, item_ids, rec_items)

    #     # self.recsys.save_round_interaction(self.round_cnt)

    #     user_reward = []
    #     self.dict_count.append(len(self.active_agents))
    #     for i in self.active_agents:
    #         item_len = self.temp_user_transition_dict[i]['recent_items'].size(0)
    #         self.temp_user_transition_dict[i]['next_interest_items'] = torch.stack([torch.tensor(self.interest_items[i]) for _ in range(item_len)])
    #         interest_items_cate = [self.data.get_cate_id_by_item(i) for i in self.interest_items[i]]
    #         self.temp_user_transition_dict[i]['next_interest_items_cate'] = torch.stack([torch.tensor(interest_items_cate) for _ in range(item_len)])
    #         self.user_transition_dict['users'].append(self.temp_user_transition_dict[i]['user'])
    #         self.user_transition_dict['interest_items'].append(self.temp_user_transition_dict[i]['interest_items'])
    #         self.user_transition_dict['interest_items_cate'].append(self.temp_user_transition_dict[i]['interest_items_cate'])
    #         self.user_transition_dict['recent_items'].append(self.temp_user_transition_dict[i]['recent_items'])
    #         self.user_transition_dict['recent_items_cate'].append(self.temp_user_transition_dict[i]['recent_items_cate'])
    #         self.user_transition_dict['actions'].append(self.temp_user_transition_dict[i]['actions'])
    #         self.user_transition_dict['next_interest_items'].append(self.temp_user_transition_dict[i]['next_interest_items'])
    #         self.user_transition_dict['next_interest_items_cate'].append(self.temp_user_transition_dict[i]['next_interest_items_cate'])
    #         self.user_transition_dict['rewards'].append(self.temp_user_transition_dict[i]['rewards'])
    #         user_reward.append(self.agents[i].return_click())
    
    #     provider_rewards = []

    #     for provider_id in self.active_proagents:
    #         proagent = self.provider_agents[provider_id]
    #         last_new_item = proagent.new_round_item[-1]
    #         last_new_item_click = proagent.item_acc_click[last_new_item]
    #         provider_rewards.append(last_new_item_click)
    #         if len(proagent.new_round_item) <=1:
    #             proagent.update_last_reward(last_new_item_click)
    #             continue
    #         proagent.update_trust(last_new_item_click)

    #         # upload_time = self.data.items[last_new_item]['upload_time']
    #         # item_age = self.round_cnt - upload_time
    #         # if item_age >= self.config['item_recency']:
    #         #     active_round =  self.config['item_recency']
    #         # else:
    #         #     active_round = item_age
    #         # item_reward_per_round = last_new_item_click/active_round

    #     if len(provider_rewards) == 0:
    #         return np.sum(user_reward), 0
    #     else:
    #         return np.sum(user_reward), np.sum(provider_rewards)

    def create_agent(self, i, api_key) -> RecAgent:
        """
        Create an agent with the given id.
        """
        LLM = utils.get_llm(config=self.config,
                            logger=self.logger,
                            api_key=api_key,
                            user='user')
        MemoryClass = (
                RecAgentMemory
            # if self.config["recagent_memory"] == "recagent"
            # else GenerativeAgentMemory
        )

        agent_memory = MemoryClass(
            llm=LLM,
            memory_retriever=self.create_new_memory_retriever(),
            now=self.now,
            verbose=False,
            reflection_threshold=10,
            embedding_model=self.embedding_model
        )
        agent = RecAgent(
            id=i,
            name=self.data.users[i]["name"],
            # age=self.data.users[i]["age"],
            # gender=self.data.users[i]["gender"],
            # traits=self.data.users[i]["traits"],
            # status=self.data.users[i]["status"],
            interest=self.data.users[i]["interest"],
            # relationships=self.data.get_relationship_names(i),
            # feature=utils.get_feature_description(self.data.users[i]["feature"]),
            memory_retriever=None, #self.create_new_memory_retriever(),
            llm=LLM,
            memory=agent_memory,
            event=reset_event(self.now),
        )
        # observations = self.data.users[i]["observations"].strip(".").split(".")
        # for observation in observations:
        #     agent.memory.add_memory(observation, now=self.now)
        return agent

    def create_provider_agent(self, i, api_key) -> ProAgent:
        """
        Create an agent with the given id.
        """
        LLM = utils.get_llm(config=self.config,
                            logger=self.logger,
                            api_key=api_key,
                            user='provider')
        MemoryClass = (
            RecAgentMemory
            # if self.config["recagent_memory"] == "recagent"
            # else GenerativeAgentMemory
        )
        
        agent_memory = MemoryClass(
            llm=LLM,
            memory_retriever=self.create_new_memory_retriever(), #utils.get_embedding_model(),
            now=self.now,
            verbose=False,
            reflection_threshold=10,
            embedding_model=self.embedding_model
        )
        items = {id: self.data.items[id] for id in self.data.providers[i]['items']}
        trust, categories_times, skill = utils.init_trust_and_skill_for_provider(items, self.config['data_name'], self.config['change_trust'], self.config['ini_trust'])
        # print(f'skillll:{skill}')
        # print(f'categories_times:{categories_times}')
        agent = ProAgent(
            id=i,
            name=self.data.providers[i]["name"],
            trust = trust,
            skill = skill,
            categories_times = categories_times,
            # status=self.data.providers[i]["mood"],
            category_history=self.data.providers[i]["category_history"],
            items=items,
            memory_retriever=None, #self.create_new_memory_retriever(),
            llm=LLM,
            memory=agent_memory,
            event=reset_event(self.now),
            config=self.config,
            active_prob=self.data.providers[i]["frequency"]
        )
        # print(self.data.items)
        # observations = self.data.users[i]["observations"].strip(".").split(".")
        # for observation in observations:
        #     agent.memory.add_memory(observation, now=self.now)
        return agent, max(skill, key=skill.get)


    def agent_creation(self):
        """
        Create agents in parallel
        """
        agents = {}
        api_keys = list(self.config["api_keys"])
        agent_num = int(self.config["agent_num"])
        # Add ONE user controllable user into the simulator if the flag is true.
        # We block the main thread when the user is creating the role.

        if self.active_method == "random":
            active_probs = [self.config["active_prob"]] * agent_num
        else:
            active_probs = np.random.pareto(self.config["active_prob"] * 10, agent_num)
            active_probs = active_probs / active_probs.max()

        for i in tqdm(range(1, agent_num+1)):
            api_key = api_keys[i % len(api_keys)]
            agent = self.create_agent(i, api_key)
            agent.active_prob = active_probs[agent.id-1]
            agents[agent.id] = agent

        return agents


    def provider_agent_creation(self):
        """
        Create  provider  agents in parallel
        """
        agents = {}
        api_keys = list(self.config["api_keys"])
        agent_num = int(self.config["provider_agent_num"])
        # Add ONE user controllable user into the simulator if the flag is true.
        # We block the main thread when the user is creating the role.

        # if self.active_method == "random":
        #     active_probs = [self.config["active_prob"]] * agent_num
        # else:
        #     active_probs = np.random.pareto(self.config["active_prob"] * 10, agent_num)
        #     active_probs = active_probs / active_probs.max()
        genre_list = []
        for i in tqdm(range(1, agent_num+1)):
            api_key = api_keys[i % len(api_keys)]
            agent, genre = self.create_provider_agent(i, api_key)
            agents[agent.id] = agent
            genre_list.append(genre)
        return agents, genre_list

    def reset(self):
        # Reset the system
        self.pause()
        self.round_cnt = 0
        log_string = ""
        self.load_simulator()
        log_string = "The system is reset, and the historic records are removed."
        self.round_msg.append(Message(agent_id=-1, action="System", content=log_string))
        return log_string


    def print_provider_information(self, signals, next_state):
        # if self.round_cnt <= 20 * 6:
        #     return
        one_index = torch.argmax(next_state, dim = 2).squeeze()
        one_index = list(one_index)
        if self.config['signal_policy'] not in ['most_popular', 'most_click', 'creator_based'] and\
        (self.config['ppo_load_checkpoint'] or (self.round_cnt > 20 * 6 and self.config['signal'])):
            for index, provider in self.provider_agents.items():
                if one_index[index - 1] == 0:
                    # print(f'provider :{provider.name}本轮没有创作内容, 对平台的信任度为{provider.trust}, 历史创作类型为:{provider.category_history.keys()}, skill为:{provider.skill}\n')
                    continue
                elif signals[index - 1] == 0:
                    self.genre_count[self.categories[int(one_index[index - 1]) - 1]] += 1
                    print(f'provider :{provider.name}对平台的信任度为{provider.trust},平台没有发送信号，{provider.name}创作的类型是:{self.categories[int(one_index[index - 1]) - 1]}，历史创作类型为:{provider.category_history.keys()}, skill为:{provider.skill}\n')
                else:
                    self.genre_count[self.categories[int(one_index[index - 1]) - 1]] += 1
                    print(f'provider :{provider.name}对平台的信任度为{provider.trust},平台推荐创作类型{self.categories[signals[index - 1] - 1]}，{provider.name}创作的类型是:{self.categories[int(one_index[index - 1]) - 1]}，历史创作类型为:{provider.category_history.keys()}, skill为:{provider.skill}\n')
        else:
            for index, provider in self.provider_agents.items():
                if one_index[index - 1] == 0:
                    # print(f'provider :{provider.name}本轮没有创作内容, 历史创作类型为:{provider.category_history.keys()}, skill为:{provider.skill}\n')
                    continue
                else:
                    self.genre_count[self.categories[int(one_index[index - 1]) - 1]] += 1
                    print(f'provider :{provider.name}创作的类型是:{self.categories[int(one_index[index - 1]) - 1]}，历史创作类型为:{provider.category_history.keys()}, skill为:{provider.skill}\n')
        total_num = 0.000001
        for _, value in self.genre_count.items():
            total_num += value
            # self.total_num += value
        for key, value in self.genre_count.items():
            print(f'第{self.round_cnt}轮，类型:{key}的内容占比为{value/total_num}')

if __name__ == '__main__':
    import wandb
    # intialize_the_simulator
    config = CfgNode(new_allowed=True)  # config/config.yaml
    config.merge_from_file('/home/chenming/mywork/study/RS/LLaMA-Factory/src/llamafactory/config/config.yaml')
    #
    logger = utils.set_logger('simulation.log', '0630')
    logger.info(f"simulator config: \n{config}")
    # logger.info(f"os.getpid()={os.getpid()}")
    simulator = Simulator(config, logger)
    wandb.init(
        project="main experiment(amazon)",
        name=f"signal to provider(DIN, popular)",
        config=config)

    state = simulator.load_simulator() # state : [1, provider_len, categories_len+1]
    simulator.play()
    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': []}
    for i in range(config['epochs']):
        episode_reward = 0
        episode_provider_reward = 0
        # transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': []}
        for step in tqdm(range(config['round']), desc='Round Processing...'):
            signals = simulator.update_round(state)
            if simulator.round_cnt <= 20:
                signals = None
            if not config['ppo_load_checkpoint'] and simulator.round_cnt <= 20 * 6:
                signals = None
            # if simulator.config['provider_decision_making'] is False:
            #     print('No Decision')
            # else:
            provider_action, next_state = simulator.provider_round(signals)

            user_reward, provider_reward = simulator.get_user_feedbacks()
            if  simulator.config['signal'] and simulator.config['signal_policy'] == 'RL':
                # if simulator.config['signal_user']:
                #     user_reward, provider_reward = simulator.get_user_feedbacks_with_signal()
                # else:
                #     user_reward, provider_reward = simulator.get_user_feedbacks()
                transition_dict['states'].append(state)
                if simulator.round_cnt <= 20:
                    transition_dict['actions'].append(torch.tensor(provider_action))
                elif not config['ppo_load_checkpoint'] and simulator.round_cnt <= 20 * 6:
                    transition_dict['actions'].append(torch.tensor(provider_action))
                else:
                    transition_dict['actions'].append(torch.tensor(signals))
                transition_dict['next_states'].append(next_state)
                # transition_dict['rewards'].append(user_reward)
                transition_dict['rewards'].append(provider_reward)
                state = next_state
                # simulator.update(transition_dict)
            # else:
            #     user_reward, provider_reward = simulator.get_user_feedbacks()
            simulator.print_provider_information(signals, next_state)
            wandb.log({
                "user_reward": user_reward,
                "provider_reward": provider_reward,
                # "total_click": total_click,
                # **genre_count,
                # **ctr_feedbacks,
                # **provider_click_dict
            }
            )
            episode_reward += user_reward
            episode_provider_reward += provider_reward
            if simulator.round_cnt % 20 == 0 and simulator.config['signal']:
                if simulator.round_cnt <= 20:
                    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': []}
                elif  simulator.config['signal_policy'] == 'RL':
                    simulator.update(transition_dict)
                    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': []}
        # if simulator.config['signal']:
        #     simulator.update(transition_dict)
            # genre_count = simulator.get_genre_item_count()
        wandb.log({
            "outer_user_reward": episode_reward,
            "outer_provider_reward": episode_provider_reward,
            # "total_click": total_click,
            # **genre_count,
            # **ctr_feedbacks,
            # **provider_click_dict
        }
        )
    with open(simulator.config['item_save_path'], 'w') as json_file:
        json.dump(simulator.data.items, json_file)
        # if simulator.round_cnt == 1:
        #     df = pd.DataFrame([provider_click_dict])
        # else:
        #     df = pd.concat([df, pd.DataFrame([provider_click_dict])], ignore_index=True)
        # df.to_csv(f'/home/xiaopeng_ye/experiment/Agent4Fairness/figures/Bandwagon_effect/provider_dict.csv')
    wandb.finish()