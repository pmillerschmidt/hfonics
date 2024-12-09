import torch
import torch.nn as nn
from typing import Tuple, List
import numpy as np


class PPOAgent:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 epsilon: float = 0.2):
        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.critic = ValueNetwork(state_dim, hidden_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select action using current policy"""
        with torch.no_grad():
            action_probs = self.actor(state)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action, log_prob

    def update(self, rollouts: List[dict]) -> Tuple[float, float]:
        """Update policy and value function using PPO"""
        # Process rollouts
        states = torch.stack([r['state'] for r in rollouts])
        actions = torch.stack([r['action'] for r in rollouts])
        old_log_probs = torch.stack([r['log_prob'] for r in rollouts])
        rewards = torch.tensor([r['reward'] for r in rollouts])

        # Compute returns
        returns = self._compute_returns(rewards)

        # Update actor
        actor_loss = self._update_actor(states, actions, old_log_probs, returns)

        # Update critic
        critic_loss = self._update_critic(states, returns)

        return actor_loss, critic_loss

    def _compute_returns(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute discounted returns"""
        returns = torch.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        return returns

    def _update_actor(self, states, actions, old_log_probs, returns) -> float:
        """Update policy network using PPO loss"""
        # TODO: Implement PPO policy update
        return 0.0

    def _update_critic(self, states, returns) -> float:
        """Update value network"""
        # TODO: Implement value function update
        return 0.0


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)