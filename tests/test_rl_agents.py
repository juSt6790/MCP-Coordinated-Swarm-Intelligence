"""Tests for RL agents functionality."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

from rl_agents.base_agent import BaseAgent, NeuralNetwork, ActorCriticNetwork
from rl_agents.ppo_agent import PPOAgent, PPOBuffer
from rl_agents.context_aware_agent import ContextAwareAgent, ContextAwareNetwork


class TestNeuralNetwork:
    """Test neural network functionality."""
    
    def test_network_creation(self):
        """Test creating a neural network."""
        network = NeuralNetwork(input_dim=10, output_dim=5, hidden_dims=[64, 32])
        
        # Test forward pass
        x = torch.randn(1, 10)
        output = network(x)
        
        assert output.shape == (1, 5)
        assert isinstance(output, torch.Tensor)
    
    def test_actor_critic_network(self):
        """Test actor-critic network."""
        network = ActorCriticNetwork(state_dim=10, action_dim=3, hidden_dims=[64, 32])
        
        # Test forward pass
        state = torch.randn(1, 10)
        action, value = network(state)
        
        assert action.shape == (1, 3)
        assert value.shape == (1, 1)
        assert torch.all(action >= -1) and torch.all(action <= 1)  # Tanh output


class TestPPOBuffer:
    """Test PPO buffer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.buffer = PPOBuffer(buffer_size=100, state_dim=10, action_dim=3)
    
    def test_add_experience(self):
        """Test adding experience to buffer."""
        state = np.random.randn(10)
        action = np.random.randn(3)
        reward = 1.0
        value = 0.5
        log_prob = -0.1
        done = False
        
        self.buffer.add(state, action, reward, value, log_prob, done)
        
        assert self.buffer.size == 1
        assert np.array_equal(self.buffer.states[0], state)
        assert np.array_equal(self.buffer.actions[0], action)
        assert self.buffer.rewards[0] == reward
    
    def test_compute_advantages(self):
        """Test computing advantages and returns."""
        # Add some test data
        for i in range(5):
            state = np.random.randn(10)
            action = np.random.randn(3)
            reward = i * 0.1
            value = i * 0.05
            log_prob = -0.1
            done = i == 4  # Last one is done
            
            self.buffer.add(state, action, reward, value, log_prob, done)
        
        next_value = 0.5
        self.buffer.compute_advantages_and_returns(next_value, gamma=0.99, lambda_gae=0.95)
        
        assert len(self.buffer.advantages) == 5
        assert len(self.buffer.returns) == 5
        assert all(not np.isnan(adv) for adv in self.buffer.advantages)
    
    def test_get_batch(self):
        """Test getting a batch from buffer."""
        # Add some test data
        for i in range(10):
            state = np.random.randn(10)
            action = np.random.randn(3)
            reward = i * 0.1
            value = i * 0.05
            log_prob = -0.1
            done = False
            
            self.buffer.add(state, action, reward, value, log_prob, done)
        
        batch = self.buffer.get_batch(5)
        
        assert "states" in batch
        assert "actions" in batch
        assert "rewards" in batch
        assert batch["states"].shape[0] == 5
        assert batch["actions"].shape[0] == 5


class TestPPOAgent:
    """Test PPO agent functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "batch_size": 64,
            "buffer_size": 2048,
            "ppo_epochs": 4,
            "epsilon_clip": 0.2,
            "value_loss_coef": 0.5,
            "entropy_coef": 0.01,
            "action_scale": 1.0
        }
        self.agent = PPOAgent("test_agent", state_dim=10, action_dim=3, config=self.config)
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        assert self.agent.agent_id == "test_agent"
        assert self.agent.state_dim == 10
        assert self.agent.action_dim == 3
        assert self.agent.device is not None
        assert self.agent.actor_critic is not None
        assert self.agent.optimizer is not None
        assert self.agent.buffer is not None
    
    def test_select_action(self):
        """Test action selection."""
        state = np.random.randn(10)
        
        # Test deterministic action
        action = self.agent.select_action(state, deterministic=True)
        assert action.shape == (3,)
        assert isinstance(action, np.ndarray)
        
        # Test stochastic action
        action = self.agent.select_action(state, deterministic=False)
        assert action.shape == (3,)
        assert isinstance(action, np.ndarray)
    
    def test_add_reward(self):
        """Test adding reward to episode."""
        initial_rewards = len(self.agent.episode_rewards)
        self.agent.add_reward(1.0)
        
        assert len(self.agent.episode_rewards) == initial_rewards + 1
        assert self.agent.episode_rewards[-1] == 1.0
        assert self.agent.episode_length == 1
    
    def test_reset_episode(self):
        """Test resetting episode."""
        self.agent.add_reward(1.0)
        self.agent.add_reward(0.5)
        
        assert len(self.agent.episode_rewards) == 2
        assert self.agent.episode_length == 2
        
        self.agent.reset_episode()
        
        assert len(self.agent.episode_rewards) == 0
        assert self.agent.episode_length == 0
    
    def test_get_episode_reward(self):
        """Test getting episode reward."""
        self.agent.add_reward(1.0)
        self.agent.add_reward(0.5)
        self.agent.add_reward(-0.2)
        
        total_reward = self.agent.get_episode_reward()
        assert total_reward == 1.3
    
    def test_update_performance_history(self):
        """Test updating performance history."""
        self.agent.add_reward(1.0)
        self.agent.add_reward(0.5)
        
        initial_history_length = len(self.agent.performance_history["rewards"])
        self.agent.update_performance_history()
        
        assert len(self.agent.performance_history["rewards"]) == initial_history_length + 1
        assert self.agent.performance_history["rewards"][-1] == 1.5
        assert self.agent.performance_history["episode_lengths"][-1] == 2
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading agent."""
        # Add some data
        self.agent.add_reward(1.0)
        self.agent.update_performance_history()
        
        # Save agent
        save_path = tmp_path / "test_agent.pth"
        self.agent.save(str(save_path))
        
        # Create new agent and load
        new_agent = PPOAgent("new_agent", state_dim=10, action_dim=3, config=self.config)
        new_agent.load(str(save_path))
        
        # Check that performance history was loaded
        assert len(new_agent.performance_history["rewards"]) == 1
        assert new_agent.performance_history["rewards"][0] == 1.0


class TestContextAwareNetwork:
    """Test context-aware network functionality."""
    
    def test_network_creation(self):
        """Test creating a context-aware network."""
        network = ContextAwareNetwork(
            state_dim=10, 
            context_dim=20, 
            action_dim=3, 
            hidden_dims=[64, 32]
        )
        
        # Test forward pass
        state = torch.randn(1, 10)
        context = torch.randn(1, 20)
        action, value = network(state, context)
        
        assert action.shape == (1, 3)
        assert value.shape == (1, 1)
        assert torch.all(action >= -1) and torch.all(action <= 1)


class TestContextAwareAgent:
    """Test context-aware agent functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "batch_size": 64,
            "buffer_size": 2048,
            "context_update_frequency": 1.0,
            "context_timeout": 5.0
        }
        self.agent = ContextAwareAgent(
            "test_agent", 
            state_dim=10, 
            action_dim=3, 
            context_dim=20, 
            config=self.config
        )
    
    def test_agent_initialization(self):
        """Test context-aware agent initialization."""
        assert self.agent.agent_id == "test_agent"
        assert self.agent.state_dim == 10
        assert self.agent.action_dim == 3
        assert self.agent.context_dim == 20
        assert isinstance(self.agent.actor_critic, ContextAwareNetwork)
    
    def test_extract_context_features(self):
        """Test extracting context features."""
        # Test with empty context
        features = self.agent._extract_context_features()
        assert features.shape == (20,)
        assert isinstance(features, np.ndarray)
        
        # Test with context data
        self.agent.context_data = {
            "coverage_map": [[1, 0, 1], [0, 1, 0], [1, 1, 0]],
            "battery_status": {"uav1": 80.0, "uav2": 60.0},
            "communication_network": {"uav1": ["uav2"]},
            "target_priorities": {"target1": 3, "target2": 2},
            "environmental_conditions": {"temperature": 25.0, "humidity": 60.0},
            "wind_conditions": {"speed": 2.0, "direction": 45.0},
            "emergency_events": []
        }
        
        features = self.agent._extract_context_features()
        assert features.shape == (20,)
        assert isinstance(features, np.ndarray)
    
    def test_get_local_coverage_features(self):
        """Test getting local coverage features."""
        coverage_map = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        features = self.agent._get_local_coverage_features(coverage_map)
        
        assert len(features) == 5
        assert all(isinstance(f, float) for f in features)
        assert 0 <= features[0] <= 1  # Average coverage
        assert features[1] >= 0  # Coverage variance
        assert 0 <= features[2] <= 1  # High coverage ratio
        assert 0 <= features[3] <= 1  # Very high coverage ratio
        assert 0 <= features[4] <= 1  # Uncovered ratio
    
    def test_get_context_usage_stats(self):
        """Test getting context usage statistics."""
        stats = self.agent.get_context_usage_stats()
        
        assert "mcp_connected" in stats
        assert "last_context_update" in stats
        assert "context_age" in stats
        assert "context_updates_received" in stats
        assert "context_updates_used" in stats
        assert "context_quality" in stats
        
        assert isinstance(stats["mcp_connected"], bool)
        assert isinstance(stats["context_updates_received"], int)
        assert isinstance(stats["context_updates_used"], int)
