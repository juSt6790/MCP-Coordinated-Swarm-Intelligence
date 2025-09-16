"""Training script for MCP-Coordinated Swarm Intelligence agents."""

import asyncio
import argparse
import numpy as np
import torch
from loguru import logger
from typing import Dict, Any, List
import time
import os

from config.simulation_config import SimulationConfig
from simulation.environment import SwarmEnvironment
from .ppo_agent import PPOAgent
from .context_aware_agent import ContextAwareAgent


class SwarmTrainer:
    """Trainer for UAV swarm coordination agents."""
    
    def __init__(self, config: SimulationConfig, use_context: bool = True):
        self.config = config
        self.use_context = use_context
        
        # Create environment
        self.env = SwarmEnvironment(config)
        
        # Create agents
        self.agents = []
        self._create_agents()
        
        # Training state
        self.episode_count = 0
        self.total_steps = 0
        self.training_metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "coverage_percentages": [],
            "battery_efficiencies": [],
            "communication_reliabilities": []
        }
        
        # Create save directory
        os.makedirs("saved_models", exist_ok=True)
    
    def _create_agents(self):
        """Create RL agents for each UAV."""
        state_dim = self.env.observation_space.shape[0] // self.config.num_uavs
        action_dim = self.env.action_space.shape[0] // self.config.num_uavs
        context_dim = 20  # Estimated context dimension
        
        for i in range(self.config.num_uavs):
            agent_id = f"uav_{i}"
            
            if self.use_context:
                agent = ContextAwareAgent(
                    agent_id=agent_id,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    context_dim=context_dim,
                    config={
                        "learning_rate": self.config.rl_config.learning_rate,
                        "gamma": self.config.rl_config.gamma,
                        "batch_size": self.config.rl_config.batch_size,
                        "buffer_size": self.config.rl_config.buffer_size,
                        "ppo_epochs": 4,
                        "epsilon_clip": 0.2,
                        "value_loss_coef": 0.5,
                        "entropy_coef": 0.01,
                        "action_scale": self.config.uav_config.max_acceleration
                    }
                )
            else:
                agent = PPOAgent(
                    agent_id=agent_id,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    config={
                        "learning_rate": self.config.rl_config.learning_rate,
                        "gamma": self.config.rl_config.gamma,
                        "batch_size": self.config.rl_config.batch_size,
                        "buffer_size": self.config.rl_config.buffer_size,
                        "ppo_epochs": 4,
                        "epsilon_clip": 0.2,
                        "value_loss_coef": 0.5,
                        "entropy_coef": 0.01,
                        "action_scale": self.config.uav_config.max_acceleration
                    }
                )
            
            self.agents.append(agent)
    
    async def train(self, num_episodes: int, save_frequency: int = 100):
        """Train the swarm agents."""
        logger.info(f"Starting training for {num_episodes} episodes")
        logger.info(f"Using context-aware agents: {self.use_context}")
        
        # Start MCP connection for context-aware agents
        if self.use_context:
            await self.env.start_mcp_connection()
            for agent in self.agents:
                if hasattr(agent, 'start_mcp_connection'):
                    await agent.start_mcp_connection()
        
        try:
            for episode in range(num_episodes):
                episode_reward, episode_length, metrics = await self._run_episode()
                
                # Update training metrics
                self.training_metrics["episode_rewards"].append(episode_reward)
                self.training_metrics["episode_lengths"].append(episode_length)
                # Metrics may be sequences over steps; store episode-level scalars
                def _to_scalar(value):
                    if isinstance(value, (list, tuple)):
                        return float(value[-1]) if len(value) > 0 else 0.0
                    try:
                        return float(value)
                    except Exception:
                        return 0.0

                    
                self.training_metrics["coverage_percentages"].append(
                    _to_scalar(metrics.get("coverage_percentage", 0.0))
                )
                self.training_metrics["battery_efficiencies"].append(
                    _to_scalar(metrics.get("battery_efficiency", 0.0))
                )
                self.training_metrics["communication_reliabilities"].append(
                    _to_scalar(metrics.get("communication_reliability", 0.0))
                )
                
                self.episode_count += 1
                
                # Log progress
                if episode % 10 == 0:
                    avg_reward = np.mean(self.training_metrics["episode_rewards"][-10:])
                    avg_length = np.mean(self.training_metrics["episode_lengths"][-10:])
                    avg_coverage = np.mean(self.training_metrics["coverage_percentages"][-10:])
                    
                    logger.info(f"Episode {episode}: Reward={avg_reward:.2f}, "
                              f"Length={avg_length:.1f}, Coverage={avg_coverage:.1f}%")
                
                # Save models
                if episode % save_frequency == 0 and episode > 0:
                    self._save_models(episode)
                
                # Update context for context-aware agents
                if self.use_context:
                    for agent in self.agents:
                        if hasattr(agent, 'update_context_async'):
                            await agent.update_context_async()
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        except Exception as e:
            logger.error(f"Training error: {e}")
        
        finally:
            # Save final models
            self._save_models("final")
            self.env.close()
            
            # Close MCP connections
            for agent in self.agents:
                if hasattr(agent, 'close'):
                    agent.close()
    
    async def _run_episode(self) -> tuple:
        """Run a single training episode."""
        # Reset environment
        observations, info = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        # Reset agents
        for agent in self.agents:
            agent.reset_episode()
        
        # Run episode
        while True:
            # Get actions from all agents
            actions = []
            for i, agent in enumerate(self.agents):
                # Extract state for this agent
                state_dim = self.env.observation_space.shape[0] // self.config.num_uavs
                start_idx = i * state_dim
                end_idx = (i + 1) * state_dim
                agent_state = observations[start_idx:end_idx]
                
                # Get action from agent
                action = agent.select_action(agent_state)
                actions.extend(action)
            
            # Step environment
            observations, reward, terminated, truncated, info = self.env.step(np.array(actions))
            
            # Update agents
            for i, agent in enumerate(self.agents):
                agent.add_reward(reward)
            
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1
            
            # Check termination
            if terminated or truncated:
                break
            
            # Update agents with experience
            if episode_length % 10 == 0:  # Update every 10 steps
                for agent in self.agents:
                    if hasattr(agent, 'update'):
                        # This is simplified - in practice, you'd need to collect
                        # proper experience tuples
                        agent.update(
                            states=np.array([]),
                            actions=np.array([]),
                            rewards=np.array([]),
                            next_states=np.array([]),
                            dones=np.array([])
                        )
        
        # Update performance history for all agents
        for agent in self.agents:
            agent.update_performance_history()
        
        return episode_reward, episode_length, info.get("performance_metrics", {})
    
    def _save_models(self, episode: str):
        """Save all agent models."""
        for i, agent in enumerate(self.agents):
            model_path = f"saved_models/agent_{i}_episode_{episode}.pth"
            agent.save(model_path)
        
        logger.info(f"Models saved for episode {episode}")
    
    def _load_models(self, episode: str):
        """Load all agent models."""
        for i, agent in enumerate(self.agents):
            model_path = f"saved_models/agent_{i}_episode_{episode}.pth"
            if os.path.exists(model_path):
                agent.load(model_path)
                logger.info(f"Loaded model for agent {i}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        if not self.training_metrics["episode_rewards"]:
            return {}
        
        return {
            "total_episodes": len(self.training_metrics["episode_rewards"]),
            "total_steps": self.total_steps,
            "average_reward": np.mean(self.training_metrics["episode_rewards"]),
            "average_episode_length": np.mean(self.training_metrics["episode_lengths"]),
            "average_coverage": np.mean(self.training_metrics["coverage_percentages"]),
            "average_battery_efficiency": np.mean(self.training_metrics["battery_efficiencies"]),
            "average_communication_reliability": np.mean(self.training_metrics["communication_reliabilities"]),
            "best_reward": np.max(self.training_metrics["episode_rewards"]),
            "recent_performance": {
                "last_10_episodes_reward": np.mean(self.training_metrics["episode_rewards"][-10:]),
                "last_10_episodes_coverage": np.mean(self.training_metrics["coverage_percentages"][-10:])
            }
        }


async def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train MCP-Coordinated Swarm Intelligence")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--no-context", action="store_true", help="Train without MCP context")
    parser.add_argument("--load-episode", type=str, help="Load models from specific episode")
    parser.add_argument("--save-freq", type=int, default=100, help="Save frequency")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = SimulationConfig.from_yaml(args.config)
    else:
        config = SimulationConfig()
    
    # Setup logging
    logger.add("logs/training.log", rotation="1 day", retention="7 days")
    os.makedirs("logs", exist_ok=True)
    
    # Create trainer
    trainer = SwarmTrainer(config, use_context=not args.no_context)
    
    # Load models if specified
    if args.load_episode:
        trainer._load_models(args.load_episode)
    
    # Start training
    await trainer.train(args.episodes, args.save_freq)
    
    # Print training summary
    summary = trainer.get_training_summary()
    logger.info("Training Summary:")
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
