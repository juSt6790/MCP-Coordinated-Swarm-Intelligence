# MCP-Coordinated Swarm Intelligence User Guide

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MCP-Coordinated-Swarm-Intelligence.git
cd MCP-Coordinated-Swarm-Intelligence

# Install Python dependencies
pip install -r requirements.txt

# Install web dashboard dependencies
cd web_dashboard
npm install
cd ..
```

### 2. Basic Usage

#### Start the MCP Server
```bash
python -m mcp_server.server
```

#### Run the Simulation
```bash
# Basic simulation
python -m simulation.main

# With custom configuration
python -m simulation.main --config config/example_simulation.yaml

# Headless mode (no visualization)
python -m simulation.main --headless
```

#### Start the Web Dashboard
```bash
cd web_dashboard
npm start
```

#### Train RL Agents
```bash
# Train context-aware agents
python3 -m rl_agents.train --episodes 1000

# Train baseline agents (no MCP)
python3 -m rl_agents.train --episodes 1000 --no-context

# With custom configuration
python3 -m rl_agents.train --config config/example_simulation.yaml --episodes 1000
```

## Configuration

### Simulation Configuration

Create a YAML configuration file or use the provided examples:

```yaml
# config/my_simulation.yaml
num_uavs: 5
simulation_time: 300.0
time_step: 0.1
render: true
render_fps: 30

uav:
  max_speed: 5.0
  max_acceleration: 2.0
  battery_capacity: 100.0
  communication_range: 50.0
  sensor_range: 30.0

environment:
  width: 1000
  height: 1000
  disaster_zones:
    - [200, 200, 100, 100]
    - [600, 400, 150, 120]
  obstacles:
    - [300, 300, 50, 50]
    - [700, 600, 80, 60]
  target_areas:
    - [100, 100, 80, 80]
    - [800, 800, 100, 100]

rl:
  algorithm: "PPO"
  learning_rate: 0.0003
  batch_size: 64
  total_timesteps: 1000000
```

### MCP Configuration

```yaml
# config/my_mcp.yaml
host: "localhost"
port: 8765
max_connections: 50
context_update_frequency: 1.0
context_retention_time: 60.0
coverage_grid_resolution: 10
battery_threshold: 20.0
```

## Running Experiments

### Baseline Comparison

Compare MCP-coordinated vs baseline performance:

```bash
python3 experiments/baseline_comparison.py
```

This will:
1. Run baseline experiments (no MCP coordination)
2. Run MCP-coordinated experiments
3. Generate performance comparison plots
4. Save results to `results/baseline_comparison.json`

### Performance Analysis

```bash
# Run performance analysis
python3 experiments/performance_analysis.py --episodes 500

# Generate detailed reports
python3 experiments/performance_analysis.py --episodes 500 --detailed
```

## Web Dashboard Usage

### Accessing the Dashboard

1. Start the web dashboard: `cd web_dashboard && npm start`
2. Open your browser to `http://localhost:3000`
3. The dashboard will automatically connect to the MCP server

### Dashboard Features

#### Main Dashboard
- **Real-time Metrics**: Live updates of swarm performance
- **UAV Status**: Individual UAV information and status
- **Coverage Visualization**: Real-time coverage progress
- **Performance Charts**: Historical performance data

#### Simulation View
- **3D Visualization**: Interactive 3D view of the simulation
- **UAV Tracking**: Real-time UAV position and trajectory
- **Environment Display**: Disaster zones, obstacles, and targets
- **Control Panel**: Start, stop, and configure simulations

#### Performance View
- **Detailed Analytics**: Comprehensive performance metrics
- **Trend Analysis**: Performance over time
- **Comparison Tools**: Compare different runs
- **Export Data**: Download performance data

#### Agent View
- **Agent Status**: Individual agent information
- **Context Usage**: MCP context utilization statistics
- **Learning Progress**: Training metrics and progress
- **Configuration**: Agent parameter settings

## Advanced Usage

### Custom Agent Development

Create custom RL agents by extending the base classes:

```python
from rl_agents.base_agent import BaseAgent

class MyCustomAgent(BaseAgent):
    def __init__(self, agent_id, state_dim, action_dim, config):
        super().__init__(agent_id, state_dim, action_dim, config)
        # Initialize your custom agent
    
    def select_action(self, state, deterministic=False):
        # Implement your action selection logic
        pass
    
    def update(self, states, actions, rewards, next_states, dones):
        # Implement your learning update logic
        pass
```

### Custom Environment Scenarios

Create custom disaster scenarios:

```python
from simulation.disaster_scenario import DisasterScenario

class MyDisasterScenario(DisasterScenario):
    def __init__(self, config):
        super().__init__(config)
        # Add your custom scenario elements
    
    def update(self, dt):
        super().update(dt)
        # Add your custom update logic
```

### MCP Context Extensions

Add custom context types to the MCP server:

```python
# In mcp_server/context_manager.py
def _get_custom_context(self) -> Dict[str, Any]:
    """Get custom context information."""
    return {
        "custom_metric": self.calculate_custom_metric(),
        "custom_data": self.get_custom_data()
    }
```

## Troubleshooting

### Common Issues

#### MCP Server Connection Failed
```
Error: Failed to connect to MCP server
```
**Solution**: Ensure the MCP server is running on the correct port (default: 8765)

#### PyGame Display Issues
```
Error: pygame.error: No available video device
```
**Solution**: Run in headless mode with `--headless` flag or install display drivers

#### Web Dashboard Not Loading
```
Error: Cannot connect to dashboard server
```
**Solution**: Ensure the dashboard server is running on port 3001 and check firewall settings

#### Training Not Converging
```
Warning: Agent performance not improving
```
**Solutions**:
- Adjust learning rate in configuration
- Increase training episodes
- Check reward function design
- Verify environment dynamics

### Performance Optimization

#### For Large Swarms (>10 UAVs)
- Increase MCP server `max_connections`
- Reduce `context_update_frequency`
- Use headless mode for training
- Increase `buffer_size` for RL agents

#### For Long Simulations (>1 hour)
- Enable data saving: `save_data: true`
- Reduce visualization frequency
- Use checkpoint saving for training
- Monitor memory usage

#### For Real-time Requirements
- Increase `render_fps` for smoother visualization
- Reduce `time_step` for higher precision
- Optimize neural network size
- Use GPU acceleration if available

## API Reference

### MCP Server API

#### Message Types
- `update`: Send context update to server
- `query`: Request context information
- `heartbeat`: Maintain connection
- `register`: Register new client

#### Context Types
- `position`: UAV position and movement
- `battery`: Battery status and health
- `sensor_data`: Sensor readings and measurements
- `coverage_map`: Area coverage information
- `communication_network`: Network topology
- `environmental_conditions`: Weather and conditions

### Simulation Environment API

#### Environment Methods
- `reset()`: Reset environment to initial state
- `step(actions)`: Execute actions and return results
- `render()`: Render current state
- `close()`: Clean up resources

#### Observation Space
- UAV position (x, y, z)
- UAV velocity (vx, vy, vz)
- Battery level
- Payload weight
- Context information (if MCP enabled)

#### Action Space
- Acceleration in x, y, z directions
- Bounded by maximum acceleration limits

### RL Agent API

#### Agent Methods
- `select_action(state)`: Choose action given state
- `update(experience)`: Update policy with experience
- `save(path)`: Save agent model
- `load(path)`: Load agent model

#### Performance Metrics
- Episode rewards
- Coverage percentage
- Battery efficiency
- Communication reliability
- Collision count

## Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest black flake8 mypy
   ```

### Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Write comprehensive docstrings
- Include unit tests for new features

### Testing

```bash
# Run all tests
pytest

# Run specific test files
pytest tests/test_mcp_server.py
pytest tests/test_rl_agents.py
pytest tests/test_simulation.py

# Run with coverage
pytest --cov=.
```

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, issues, or contributions:

1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed description
4. Contact the development team

## Acknowledgments

- PyGame for simulation framework
- Stable-Baselines3 for RL algorithms
- React.js for web dashboard
- PyTorch for neural networks
- The open-source community for inspiration and tools
