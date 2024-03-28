import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

import random
from collections import namedtuple, deque
import numpy as np
import torch.nn.functional as F

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(ActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.conv = nn.Sequential(
            nn.Conv2d(state_size[0], 64, kernel_size=8, stride=4),
            nn.Tanh(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.Tanh(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.Tanh()
            )
        
        self.fc = nn.Sequential(
            nn.Linear(128*7*7, 1024),
            nn.Tanh(),
            nn.Linear(1024, 256),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_size), std=0.01), 
            nn.Softmax(dim=-1)
        )
        
    def forward(self, state):
        conv_out = self.conv(state)
        # print(conv_out.shape)
        conv_out = conv_out.view(-1, 128*7*7)
        return self.fc(conv_out)
        
class CriticNetwork(nn.Module):
    def __init__(self, state_size, seed):
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.conv = nn.Sequential(
            nn.Conv2d(state_size[0], 64, kernel_size=8, stride=4),
            nn.Tanh(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.Tanh(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.Tanh()
            )
        
        self.fc = nn.Sequential(
            nn.Linear(128*7*7, 1024),
            nn.Tanh(),
            nn.Linear(1024, 256),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0)
        )
        
    def forward(self, state):   
        conv_out = self.conv(state)
        # print(conv_out.shape)
        conv_out = conv_out.view(-1, 128*7*7)
        return self.fc(conv_out)
    
class PPOPolicy(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(PPOPolicy, self).__init__()
        self.actor = ActorNetwork(state_size, action_size, seed).to(device)
        self.critic = CriticNetwork(state_size, seed).to(device)
        
    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_value = self.critic(state)
        
        return action.detach(), action_logprob.detach(), state_value.detach()
    
    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy
    
    def evaluate_state_value(self, state):
        return self.critic(state)
    
    def forward(self):
        raise NotImplementedError
    
    
class PPOAgent:
    def __init__(self, state_size, action_size, lr_actor, lr_critic, gamma, K_epochs, eps_clip, seed):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()
        
        self.policy = PPOPolicy(state_size, action_size, seed).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        
        self.policy_old = PPOPolicy(state_size, action_size, seed).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def compute_gae(self, next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns
        
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_value = self.policy_old.act(state)
            
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_value)
        
        return action.item()
    
    def update(self):
        # not use GAE(方法1)
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards   
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # use GAE to compute returns(方法2，理论来说性能更好，但是实际上效果不收敛)
        # with torch.no_grad():
        #     next_state_value = self.policy.evaluate_state_value(self.buffer.states[-1])

        # # Compute GAE
        # returns = self.compute_gae(next_state_value, self.buffer.rewards, self.buffer.is_terminals, self.buffer.state_values)

        # # Normalizing the returns
        # returns = torch.tensor(returns).to(device)
        # returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        # old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)
        
        advantages = rewards - torch.tensor(self.buffer.state_values).to(device)
        
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5) #梯度裁剪
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.buffer.clear()
        
    def save(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path)
        
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location = lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location = lambda storage, loc: storage))

