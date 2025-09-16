"""Base agent class for UAV swarm reinforcement learning."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn


class BaseAgent(ABC):
    """Base class for all RL agents in the swarm."""
    
    def __init__(self, agent_id: str, state_dim: int, action_dim: int, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Agent state
        self.episode_rewards = []
        self.episode_length = 0
        self.total_steps = 0
        
        # Performance tracking
        self.performance_history = {
            "rewards": [],
            "episode_lengths": [],
            "exploration_rate": [],
            "losses": []
        }
    
    @abstractmethod
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select an action given the current state."""
        pass
    
    @abstractmethod
    def update(self, states: np.ndarray, actions: np.ndarray, 
               rewards: np.ndarray, next_states: np.ndarray, 
               dones: np.ndarray) -> Dict[str, float]:
        """Update the agent with new experience."""
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save the agent's model."""
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load the agent's model."""
        pass
    
    def reset_episode(self):
        """Reset agent state for new episode."""
        self.episode_rewards = []
        self.episode_length = 0
    
    def add_reward(self, reward: float):
        """Add reward to current episode."""
        self.episode_rewards.append(reward)
        self.episode_length += 1
        self.total_steps += 1
    
    def get_episode_reward(self) -> float:
        """Get total reward for current episode."""
        return sum(self.episode_rewards)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the agent."""
        if not self.performance_history["rewards"]:
            return {}
        
        return {
            "mean_reward": np.mean(self.performance_history["rewards"][-100:]),
            "mean_episode_length": np.mean(self.performance_history["episode_lengths"][-100:]),
            "total_episodes": len(self.performance_history["rewards"]),
            "total_steps": self.total_steps,
            "agent_id": self.agent_id
        }
    
    def update_performance_history(self):
        """Update performance history with current episode data."""
        if self.episode_rewards:
            self.performance_history["rewards"].append(self.get_episode_reward())
            self.performance_history["episode_lengths"].append(self.episode_length)
    
    def set_training_mode(self, training: bool):
        """Set training mode for the agent."""
        pass  # Override in subclasses if needed
    
    def get_action_space_info(self) -> Dict[str, Any]:
        """Get information about the action space."""
        return {
            "action_dim": self.action_dim,
            "action_bounds": (-1.0, 1.0),  # Normalized actions
            "agent_id": self.agent_id
        }
    
    def get_state_space_info(self) -> Dict[str, Any]:
        """Get information about the state space."""
        return {
            "state_dim": self.state_dim,
            "agent_id": self.agent_id
        }


class NeuralNetwork(nn.Module):
    """Base neural network for RL agents."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for policy gradient methods."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()
        
        # Shared layers
        shared_layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.shared_network = nn.Sequential(*shared_layers)
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(prev_dim, action_dim),
            nn.Tanh()  # Bounded actions
        )
        
        # Critic head (value function)
        self.critic = nn.Linear(prev_dim, 1)
    
    def forward(self, state):
        shared_features = self.shared_network(state)
        action = self.actor(shared_features)
        value = self.critic(shared_features)
        return action, value
    
    def get_action_and_value(self, state, action=None):
        """Get action and value for given state."""
        action_mean = self.actor(self.shared_network(state))
        value = self.critic(self.shared_network(state))
        
        if action is None:
            # Sample action from policy
            action_std = torch.ones_like(action_mean) * 0.1  # Fixed std for simplicity
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        else:
            # Evaluate given action
            action_std = torch.ones_like(action_mean) * 0.1
            dist = torch.distributions.Normal(action_mean, action_std)
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value
