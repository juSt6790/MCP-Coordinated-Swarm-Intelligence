"""Baseline comparison experiment for MCP-Coordinated Swarm Intelligence."""

import asyncio
import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import os
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
import json
import time
from typing import Dict, List, Any

from config.simulation_config import SimulationConfig
from simulation.environment import SwarmEnvironment
from rl_agents.ppo_agent import PPOAgent
from rl_agents.context_aware_agent import ContextAwareAgent


class BaselineComparison:
    """Compare MCP-coordinated vs baseline swarm performance."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.results = {
            "baseline": [],
            "mcp_coordinated": []
        }
    
    async def run_baseline_experiment(self, num_episodes: int = 100) -> List[Dict[str, Any]]:
        """Run baseline experiment without MCP coordination."""
        logger.info("Running baseline experiment (no MCP coordination)")
        
        # Create environment without MCP
        env = SwarmEnvironment(self.config)
        
        # Create baseline agents
        agents = self._create_baseline_agents(env)
        
        episode_results = []
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                # Get actions from agents
                actions = self._get_agent_actions(agents, obs)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(actions)
                
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            # Record episode results
            episode_result = {
                "episode": episode,
                "reward": episode_reward,
                "length": episode_length,
                "coverage": info["performance_metrics"]["coverage_percentage"][-1] if info["performance_metrics"]["coverage_percentage"] else 0,
                "battery_efficiency": np.mean(info["performance_metrics"]["battery_efficiency"]) if info["performance_metrics"]["battery_efficiency"] else 0,
                "communication_reliability": np.mean(info["performance_metrics"]["communication_reliability"]) if info["performance_metrics"]["communication_reliability"] else 0,
                "collision_count": info["performance_metrics"]["collision_count"],
                "mission_success": info["performance_metrics"]["mission_success"]
            }
            
            episode_results.append(episode_result)
            
            if episode % 10 == 0:
                logger.info(f"Baseline Episode {episode}: Reward={episode_reward:.2f}, "
                          f"Coverage={episode_result['coverage']:.1f}%")
        
        env.close()
        return episode_results
    
    async def run_mcp_experiment(self, num_episodes: int = 100) -> List[Dict[str, Any]]:
        """Run MCP-coordinated experiment."""
        logger.info("Running MCP-coordinated experiment")
        
        # Create environment with MCP
        env = SwarmEnvironment(self.config)
        await env.start_mcp_connection()
        
        # Create MCP-coordinated agents
        agents = self._create_mcp_agents(env)
        
        # Start MCP connections for agents
        for agent in agents:
            if hasattr(agent, 'start_mcp_connection'):
                await agent.start_mcp_connection()
        
        episode_results = []
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                # Get actions from agents
                actions = self._get_agent_actions(agents, obs)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(actions)
                
                episode_reward += reward
                episode_length += 1
                
                # Update context for MCP agents
                for agent in agents:
                    if hasattr(agent, 'update_context_async'):
                        await agent.update_context_async()
                
                if terminated or truncated:
                    break
            
            # Record episode results
            episode_result = {
                "episode": episode,
                "reward": episode_reward,
                "length": episode_length,
                "coverage": info["performance_metrics"]["coverage_percentage"][-1] if info["performance_metrics"]["coverage_percentage"] else 0,
                "battery_efficiency": np.mean(info["performance_metrics"]["battery_efficiency"]) if info["performance_metrics"]["battery_efficiency"] else 0,
                "communication_reliability": np.mean(info["performance_metrics"]["communication_reliability"]) if info["performance_metrics"]["communication_reliability"] else 0,
                "collision_count": info["performance_metrics"]["collision_count"],
                "mission_success": info["performance_metrics"]["mission_success"]
            }
            
            episode_results.append(episode_result)
            
            if episode % 10 == 0:
                logger.info(f"MCP Episode {episode}: Reward={episode_reward:.2f}, "
                          f"Coverage={episode_result['coverage']:.1f}%")
        
        # Close connections
        env.close()
        for agent in agents:
            if hasattr(agent, 'close'):
                agent.close()
        
        return episode_results
    
    def _create_baseline_agents(self, env: SwarmEnvironment) -> List[PPOAgent]:
        """Create baseline PPO agents without MCP coordination."""
        agents = []
        state_dim = env.observation_space.shape[0] // self.config.num_uavs
        action_dim = env.action_space.shape[0] // self.config.num_uavs
        
        for i in range(self.config.num_uavs):
            agent = PPOAgent(
                agent_id=f"baseline_uav_{i}",
                state_dim=state_dim,
                action_dim=action_dim,
                config={
                    "learning_rate": self.config.rl_config.learning_rate,
                    "gamma": self.config.rl_config.gamma,
                    "batch_size": self.config.rl_config.batch_size,
                    "buffer_size": self.config.rl_config.buffer_size,
                    "action_scale": self.config.uav_config.max_acceleration
                }
            )
            agents.append(agent)
        
        return agents
    
    def _create_mcp_agents(self, env: SwarmEnvironment) -> List[ContextAwareAgent]:
        """Create MCP-coordinated agents."""
        agents = []
        state_dim = env.observation_space.shape[0] // self.config.num_uavs
        action_dim = env.action_space.shape[0] // self.config.num_uavs
        context_dim = 20
        
        for i in range(self.config.num_uavs):
            agent = ContextAwareAgent(
                agent_id=f"mcp_uav_{i}",
                state_dim=state_dim,
                action_dim=action_dim,
                context_dim=context_dim,
                config={
                    "learning_rate": self.config.rl_config.learning_rate,
                    "gamma": self.config.rl_config.gamma,
                    "batch_size": self.config.rl_config.batch_size,
                    "buffer_size": self.config.rl_config.buffer_size,
                    "action_scale": self.config.uav_config.max_acceleration
                }
            )
            agents.append(agent)
        
        return agents
    
    def _get_agent_actions(self, agents: List, observations: np.ndarray) -> np.ndarray:
        """Get actions from all agents."""
        actions = []
        state_dim = len(observations) // len(agents)
        
        for i, agent in enumerate(agents):
            start_idx = i * state_dim
            end_idx = (i + 1) * state_dim
            agent_obs = observations[start_idx:end_idx]
            
            action = agent.select_action(agent_obs)
            actions.extend(action)
        
        return np.array(actions)
    
    def analyze_results(self, baseline_results: List[Dict], mcp_results: List[Dict]) -> Dict[str, Any]:
        """Analyze and compare results."""
        analysis = {
            "baseline_stats": self._calculate_stats(baseline_results),
            "mcp_stats": self._calculate_stats(mcp_results),
            "improvements": {}
        }
        
        # Calculate improvements
        for metric in ["reward", "coverage", "battery_efficiency", "communication_reliability"]:
            baseline_avg = analysis["baseline_stats"][f"avg_{metric}"]
            mcp_avg = analysis["mcp_stats"][f"avg_{metric}"]
            
            if baseline_avg != 0:
                improvement = ((mcp_avg - baseline_avg) / baseline_avg) * 100
            else:
                improvement = 0
            
            analysis["improvements"][metric] = {
                "baseline": baseline_avg,
                "mcp": mcp_avg,
                "improvement_percent": improvement
            }
        
        return analysis
    
    def _calculate_stats(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate statistics for a set of results."""
        if not results:
            return {}
        
        metrics = ["reward", "coverage", "battery_efficiency", "communication_reliability"]
        stats = {}
        
        for metric in metrics:
            values = [r[metric] for r in results if metric in r]
            if values:
                stats[f"avg_{metric}"] = np.mean(values)
                stats[f"std_{metric}"] = np.std(values)
                stats[f"min_{metric}"] = np.min(values)
                stats[f"max_{metric}"] = np.max(values)
        
        # Mission success rate
        success_count = sum(1 for r in results if r.get("mission_success", False))
        stats["mission_success_rate"] = success_count / len(results) * 100
        
        # Collision rate
        total_collisions = sum(r.get("collision_count", 0) for r in results)
        stats["avg_collisions_per_episode"] = total_collisions / len(results)
        
        return stats
    
    def plot_results(self, baseline_results: List[Dict], mcp_results: List[Dict], save_path: str = None):
        """Plot comparison results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract data
        baseline_episodes = [r["episode"] for r in baseline_results]
        baseline_rewards = [r["reward"] for r in baseline_results]
        baseline_coverage = [r["coverage"] for r in baseline_results]
        
        mcp_episodes = [r["episode"] for r in mcp_results]
        mcp_rewards = [r["reward"] for r in mcp_results]
        mcp_coverage = [r["coverage"] for r in mcp_results]
        
        # Plot rewards
        axes[0, 0].plot(baseline_episodes, baseline_rewards, label="Baseline", alpha=0.7)
        axes[0, 0].plot(mcp_episodes, mcp_rewards, label="MCP-Coordinated", alpha=0.7)
        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot coverage
        axes[0, 1].plot(baseline_episodes, baseline_coverage, label="Baseline", alpha=0.7)
        axes[0, 1].plot(mcp_episodes, mcp_coverage, label="MCP-Coordinated", alpha=0.7)
        axes[0, 1].set_title("Coverage Percentage")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Coverage (%)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot moving averages
        window = 10
        baseline_rewards_ma = np.convolve(baseline_rewards, np.ones(window)/window, mode='valid')
        mcp_rewards_ma = np.convolve(mcp_rewards, np.ones(window)/window, mode='valid')
        
        axes[1, 0].plot(baseline_episodes[window-1:], baseline_rewards_ma, label="Baseline MA", linewidth=2)
        axes[1, 0].plot(mcp_episodes[window-1:], mcp_rewards_ma, label="MCP-Coordinated MA", linewidth=2)
        axes[1, 0].set_title(f"Moving Average Rewards (window={window})")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Reward")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot coverage moving averages
        baseline_coverage_ma = np.convolve(baseline_coverage, np.ones(window)/window, mode='valid')
        mcp_coverage_ma = np.convolve(mcp_coverage, np.ones(window)/window, mode='valid')
        
        axes[1, 1].plot(baseline_episodes[window-1:], baseline_coverage_ma, label="Baseline MA", linewidth=2)
        axes[1, 1].plot(mcp_episodes[window-1:], mcp_coverage_ma, label="MCP-Coordinated MA", linewidth=2)
        axes[1, 1].set_title(f"Moving Average Coverage (window={window})")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Coverage (%)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Results plot saved to {save_path}")
        
        plt.show()
    
    async def run_comparison(self, num_episodes: int = 100, save_results: bool = True):
        """Run complete comparison experiment."""
        logger.info(f"Starting baseline comparison experiment with {num_episodes} episodes each")
        
        # Run baseline experiment
        baseline_results = await self.run_baseline_experiment(num_episodes)
        
        # Run MCP experiment
        mcp_results = await self.run_mcp_experiment(num_episodes)
        
        # Analyze results
        analysis = self.analyze_results(baseline_results, mcp_results)
        
        # Print results
        self._print_analysis(analysis)
        
        # Plot results
        self.plot_results(baseline_results, mcp_results, "results/baseline_comparison.png")
        
        # Save results
        if save_results:
            os.makedirs("results", exist_ok=True)
            results_data = {
                "baseline_results": baseline_results,
                "mcp_results": mcp_results,
                "analysis": analysis,
                "config": {
                    "num_episodes": num_episodes,
                    "num_uavs": self.config.num_uavs,
                    "simulation_time": self.config.simulation_time
                }
            }
            
            with open("results/baseline_comparison.json", "w") as f:
                json.dump(results_data, f, indent=2)
            
            logger.info("Results saved to results/baseline_comparison.json")
        
        return analysis
    
    def _print_analysis(self, analysis: Dict[str, Any]):
        """Print analysis results."""
        print("\n" + "="*60)
        print("BASELINE COMPARISON RESULTS")
        print("="*60)
        
        print("\nBASELINE PERFORMANCE:")
        baseline_stats = analysis["baseline_stats"]
        for metric, value in baseline_stats.items():
            print(f"  {metric}: {value:.3f}")
        
        print("\nMCP-COORDINATED PERFORMANCE:")
        mcp_stats = analysis["mcp_stats"]
        for metric, value in mcp_stats.items():
            print(f"  {metric}: {value:.3f}")
        
        print("\nIMPROVEMENTS:")
        improvements = analysis["improvements"]
        for metric, data in improvements.items():
            print(f"  {metric}:")
            print(f"    Baseline: {data['baseline']:.3f}")
            print(f"    MCP: {data['mcp']:.3f}")
            print(f"    Improvement: {data['improvement_percent']:.1f}%")
        
        print("\n" + "="*60)


async def main():
    """Main function to run baseline comparison."""
    # Load configuration
    config = SimulationConfig()
    config.num_uavs = 3
    config.simulation_time = 60.0  # Shorter for testing
    config.render = False  # Disable rendering for experiments
    
    # Create comparison experiment
    comparison = BaselineComparison(config)
    
    # Run comparison
    analysis = await comparison.run_comparison(num_episodes=50)
    
    return analysis


if __name__ == "__main__":
    asyncio.run(main())
