#!/usr/bin/env python3
"""Test script to verify MCP-Coordinated Swarm Intelligence installation."""

import sys
import traceback
from loguru import logger

def test_imports():
    """Test that all required modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        # Core dependencies
        import numpy as np
        import scipy
        import matplotlib
        import pygame
        logger.info("✓ Core dependencies imported")
        
        # ML/AI dependencies
        import torch
        import stable_baselines3
        import gymnasium
        logger.info("✓ ML/AI dependencies imported")
        
        # Web dependencies
        import websockets
        import flask
        logger.info("✓ Web dependencies imported")
        
        # Project modules
        from config.simulation_config import SimulationConfig
        from config.mcp_config import MCPConfig
        logger.info("✓ Configuration modules imported")
        
        from mcp_server.server import MCPServer
        from mcp_server.context_manager import ContextManager
        from mcp_server.message_protocol import MessageProtocol
        logger.info("✓ MCP server modules imported")
        
        from simulation.environment import SwarmEnvironment
        from simulation.uav import UAV
        from simulation.disaster_scenario import DisasterScenario
        logger.info("✓ Simulation modules imported")
        
        from rl_agents.ppo_agent import PPOAgent
        from rl_agents.context_aware_agent import ContextAwareAgent
        logger.info("✓ RL agent modules imported")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration loading."""
    logger.info("Testing configuration...")
    
    try:
        from config.simulation_config import SimulationConfig
        from config.mcp_config import MCPConfig
        
        # Test simulation config
        sim_config = SimulationConfig()
        assert sim_config.num_uavs > 0
        assert sim_config.simulation_time > 0
        logger.info("✓ Simulation configuration created")
        
        # Test MCP config
        mcp_config = MCPConfig()
        assert mcp_config.port > 0
        assert mcp_config.host is not None
        logger.info("✓ MCP configuration created")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        return False

def test_simulation():
    """Test simulation environment."""
    logger.info("Testing simulation environment...")
    
    try:
        from config.simulation_config import SimulationConfig
        from simulation.environment import SwarmEnvironment
        
        # Create minimal config for testing
        config = SimulationConfig()
        config.num_uavs = 2
        config.render = False
        config.simulation_time = 1.0
        
        # Create environment
        env = SwarmEnvironment(config)
        assert len(env.uavs) == 2
        logger.info("✓ Environment created with 2 UAVs")
        
        # Test reset
        obs, info = env.reset()
        assert obs.shape[0] > 0
        logger.info(f"✓ Environment reset successful, observation shape: {obs.shape}")
        
        # Test step
        actions = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(actions)
        assert isinstance(reward, (int, float))
        logger.info("✓ Environment step successful")
        
        # Cleanup
        env.close()
        logger.info("✓ Environment closed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Simulation test failed: {e}")
        traceback.print_exc()
        return False

def test_mcp_server():
    """Test MCP server components."""
    logger.info("Testing MCP server components...")
    
    try:
        from config.mcp_config import MCPConfig, ContextMessage
        from mcp_server.context_manager import ContextManager
        from mcp_server.message_protocol import MessageProtocol
        
        # Test MCP config
        config = MCPConfig()
        logger.info("✓ MCP config created")
        
        # Test context manager
        context_manager = ContextManager(config)
        assert context_manager.coverage_grid is not None
        logger.info("✓ Context manager created")
        
        # Test message protocol
        protocol = MessageProtocol()
        logger.info("✓ Message protocol created")
        
        # Test context message
        message = ContextMessage(
            message_type="test",
            sender_id="test_agent",
            timestamp=1234567890.0,
            context_type="test",
            data={"test": "data"}
        )
        assert message.message_type == "test"
        logger.info("✓ Context message created")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ MCP server test failed: {e}")
        traceback.print_exc()
        return False

def test_rl_agents():
    """Test RL agent components."""
    logger.info("Testing RL agent components...")
    
    try:
        from rl_agents.ppo_agent import PPOAgent
        from rl_agents.context_aware_agent import ContextAwareAgent
        
        # Test PPO agent
        ppo_agent = PPOAgent(
            agent_id="test_agent",
            state_dim=10,
            action_dim=3,
            config={"learning_rate": 0.001}
        )
        assert ppo_agent.agent_id == "test_agent"
        logger.info("✓ PPO agent created")
        
        # Test context-aware agent
        context_agent = ContextAwareAgent(
            agent_id="test_context_agent",
            state_dim=10,
            action_dim=3,
            context_dim=20,
            config={"learning_rate": 0.001}
        )
        assert context_agent.agent_id == "test_context_agent"
        logger.info("✓ Context-aware agent created")
        
        # Test action selection
        state = [0.1] * 10
        action = ppo_agent.select_action(state, deterministic=True)
        assert len(action) == 3
        logger.info("✓ Action selection works")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ RL agent test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("MCP-Coordinated Swarm Intelligence Installation Test")
    logger.info("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configuration),
        ("Simulation Test", test_simulation),
        ("MCP Server Test", test_mcp_server),
        ("RL Agents Test", test_rl_agents),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Installation is successful.")
        logger.info("\nYou can now run the demo with:")
        logger.info("  python3 run_demo.py --demo simulation --duration 30")
        logger.info("  python3 run_demo.py --demo full --duration 60")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
