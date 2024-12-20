import random
from collections import deque, namedtuple
from copy import deepcopy


import numpy
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch_ac.algos.base import BaseAlgo
from torch_ac.utils import DictList


class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.Transition = namedtuple('Transition',
                                     ['state', 'action', 'reward', 'state_', 'done'])

    def push(self, *args):
        self.buffer.append(self.Transition(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)

        states_obs = [trans.state.image for trans in transitions]
        states_text = [trans.state.text for trans in transitions]
        actions = [trans.action for trans in transitions]
        rewards = [trans.reward for trans in transitions]
        next_states_obs = [trans.state_.image for trans in transitions]
        next_states_text = [trans.state_.text for trans in transitions]
        dones = [trans.done for trans in transitions]

        state_obs_tensor = torch.stack(states_obs)
        state_text_tensor = torch.stack(states_text)
        state = DictList({"image": state_obs_tensor,
                          "text": state_text_tensor})
        action_tensor = torch.stack(actions)
        reward_tensor = torch.stack(rewards)
        next_state_obs_tensor = torch.stack(next_states_obs)
        next_state_text_tensor = torch.stack(next_states_text)
        next_state = DictList({"image": next_state_obs_tensor,
                               "text": next_state_text_tensor})
        done_tensor = torch.stack(dones)

        return state, action_tensor, reward_tensor, next_state, done_tensor

    def __len__(self):
        return len(self.buffer)


class DQNAlgo(BaseAlgo):
    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99,
                lr=0.01, max_grad_norm=None, adam_eps=1e-8, epochs=4, buffer_size=10000, batch_size=32,
                target_update=10, preprocess_obss=None, reshape_reward=None,trained = False):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr,
                        0.95, 0.01, 0.5, max_grad_norm, 1, preprocess_obss, reshape_reward)

        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        self.epochs = epochs
        self.batch_size = batch_size
        self.target_update = target_update
        self.buffer_size = buffer_size
        self.optimizer = optim.Adam(self.acmodel.parameters(), lr=lr, eps=adam_eps)
        self.steps_done = 0

        self.target_model = deepcopy(acmodel)
        self.target_model.eval()
        self.trained = trained
        
    def select_action(self, state, epsilon):
        sample = random.random()
        if sample > epsilon:
            with torch.no_grad():
                action =  self.acmodel(state).max(1)[1].view(1, 1)
        else:
            action =  torch.tensor([[random.randrange(self.env.action_space.n)]], device=self.device, dtype=torch.long)

        # print(action)
        return action

    def update_target(self):
        self.target_model.load_state_dict(self.acmodel.state_dict())

    def optimize_model(self, s, a, r, s_):
        state_batch = s
        action_batch = a
        reward_batch = r
        next_states_batch = s_

        action_batch = torch.tensor(action_batch, device=self.device, dtype=torch.int64)
        action_batch = action_batch.unsqueeze(1)
        
        # 当前Q值
        state_action_values = self.acmodel(state_batch).gather(1, action_batch)

        # Double DQN: 用当前网络选择动作,用目标网络评估动作
        with torch.no_grad():
            # 用当前网络选择动作
            next_state_actions = self.acmodel(next_states_batch).max(1)[1].unsqueeze(1)
            # 用目标网络评估这些动作的Q值
            next_state_values = self.target_model(next_states_batch).gather(1, next_state_actions)

        expected_state_action_values = (next_state_values * self.discount) + reward_batch.unsqueeze(1)

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
        self.optimizer.step()

        grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters() if p.grad is not None) ** 0.5

        return loss.item(), grad_norm, state_action_values.mean().item()

    def update_parameters(self, exps):
        # Collect experiences
        for i in range(len(exps)):
            self.replay_buffer.push(exps.obs[i], exps.action[i], exps.reward[i], exps.obs_[i], exps.mask[i])
        log_losses = []
        log_grad_norms = []
        log_q_values = []
        for _ in range(self.epochs):
            if len(self.replay_buffer) < self.batch_size:
                continue
            if len(self.replay_buffer) < self.buffer_size / 4:
                print(len(self.replay_buffer))
                print("replay buffer is not full")
                continue
            

            s, a, r, s_, d = self.replay_buffer.sample(self.batch_size)
            loss, grad_norm, q_value = self.optimize_model(s, a, r, s_)
            if self.steps_done % self.target_update == 0:
                self.update_target()
            self.steps_done += 1

            log_losses.append(loss)
            log_grad_norms.append(grad_norm)
            log_q_values.append(q_value)

        logs = {
            "loss": numpy.mean(log_losses),
            "grad_norm": numpy.mean(log_grad_norms),
            "q_value": numpy.mean(log_q_values)
        }

        return logs
