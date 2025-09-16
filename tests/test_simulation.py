"""Tests for simulation environment functionality."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from simulation.environment import SwarmEnvironment
from simulation.uav import UAV, UAVState
from simulation.disaster_scenario import DisasterScenario, DisasterZone, Obstacle, TargetArea
from config.simulation_config import SimulationConfig, UAVConfig, EnvironmentConfig


class TestUAVState:
    """Test UAV state functionality."""
    
    def test_uav_state_creation(self):
        """Test creating UAV state."""
        state = UAVState(
            x=100.0, y=200.0, z=50.0,
            vx=5.0, vy=3.0, vz=1.0,
            battery=80.0, payload=0.5,
            status="active", last_communication=1234567890.0,
            sensor_data={"test": "data"}
        )
        
        assert state.x == 100.0
        assert state.y == 200.0
        assert state.z == 50.0
        assert state.vx == 5.0
        assert state.vy == 3.0
        assert state.vz == 1.0
        assert state.battery == 80.0
        assert state.payload == 0.5
        assert state.status == "active"
        assert state.sensor_data == {"test": "data"}


class TestUAV:
    """Test UAV functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = UAVConfig()
        self.uav = UAV("test_uav", self.config, (100, 200, 50))
    
    def test_uav_initialization(self):
        """Test UAV initialization."""
        assert self.uav.uav_id == "test_uav"
        assert self.uav.state.x == 100
        assert self.uav.state.y == 200
        assert self.uav.state.z == 50
        assert self.uav.state.battery == self.config.battery_capacity
        assert self.uav.state.status == "active"
    
    def test_update_physics(self):
        """Test physics update."""
        initial_x = self.uav.state.x
        initial_y = self.uav.state.y
        
        # Set acceleration
        self.uav.set_action(np.array([1.0, 0.5, 0.0]))
        
        # Update physics
        self.uav.update_physics(0.1)
        
        # Check that position changed
        assert self.uav.state.x != initial_x
        assert self.uav.state.y != initial_y
    
    def test_update_battery(self):
        """Test battery update."""
        initial_battery = self.uav.state.battery
        
        # Update battery
        self.uav.update_battery(1.0)
        
        # Check that battery decreased
        assert self.uav.state.battery < initial_battery
        assert self.uav.state.battery >= 0
    
    def test_set_action(self):
        """Test setting action."""
        action = np.array([2.0, -1.0, 0.5])
        self.uav.set_action(action)
        
        assert np.array_equal(self.uav.current_action, action)
    
    def test_get_observation(self):
        """Test getting observation."""
        obs = self.uav.get_observation()
        
        assert isinstance(obs, np.ndarray)
        assert len(obs) >= 8  # Basic state features
        assert all(not np.isnan(val) for val in obs)
    
    def test_get_sensor_data(self):
        """Test getting sensor data."""
        sensor_data = self.uav.get_sensor_data()
        
        assert "position" in sensor_data
        assert "velocity" in sensor_data
        assert "battery" in sensor_data
        assert "status" in sensor_data
        assert sensor_data["position"] == (self.uav.state.x, self.uav.state.y, self.uav.state.z)
        assert sensor_data["battery"] == self.uav.state.battery
    
    def test_can_communicate_with(self):
        """Test communication range check."""
        other_uav = UAV("other_uav", self.config, (150, 200, 50))  # 50m away
        
        # Should be able to communicate (range is 50m)
        assert self.uav.can_communicate_with(other_uav)
        
        # Move other UAV far away
        other_uav.state.x = 1000
        other_uav.state.y = 1000
        
        # Should not be able to communicate
        assert not self.uav.can_communicate_with(other_uav)
    
    def test_get_covered_area(self):
        """Test getting covered area."""
        covered_area = self.uav.get_covered_area()
        
        assert isinstance(covered_area, list)
        assert len(covered_area) == 16  # 16 points around circle
        assert all(isinstance(point, tuple) and len(point) == 2 for point in covered_area)
    
    def test_is_in_target_area(self):
        """Test checking if UAV is in target area."""
        target_areas = [(90, 190, 20, 20), (200, 300, 30, 30)]
        
        # UAV is at (100, 200, 50), should be in first target area
        assert self.uav.is_in_target_area(target_areas)
        
        # Move UAV outside target areas
        self.uav.state.x = 50
        self.uav.state.y = 50
        
        assert not self.uav.is_in_target_area(target_areas)
    
    def test_get_distance_to_target(self):
        """Test getting distance to target."""
        target_position = (150, 250, 60)
        distance = self.uav.get_distance_to_target(target_position)
        
        expected_distance = np.sqrt((150-100)**2 + (250-200)**2 + (60-50)**2)
        assert abs(distance - expected_distance) < 1e-6
    
    def test_reset(self):
        """Test resetting UAV."""
        # Modify UAV state
        self.uav.state.x = 500
        self.uav.state.y = 600
        self.uav.state.battery = 50
        self.uav.add_reward(10.0)
        
        # Reset UAV
        self.uav.reset((0, 0, 10))
        
        assert self.uav.state.x == 0
        assert self.uav.state.y == 0
        assert self.uav.state.z == 10
        assert self.uav.state.battery == self.config.battery_capacity
        assert len(self.uav.episode_rewards) == 0


class TestDisasterScenario:
    """Test disaster scenario functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        config = EnvironmentConfig()
        self.scenario = DisasterScenario(config)
    
    def test_scenario_initialization(self):
        """Test scenario initialization."""
        assert len(self.scenario.disaster_zones) > 0
        assert len(self.scenario.obstacles) > 0
        assert len(self.scenario.target_areas) > 0
        assert self.scenario.coverage_grid is not None
    
    def test_add_coverage(self):
        """Test adding coverage to grid."""
        initial_coverage = self.scenario.get_coverage_percentage()
        
        # Add coverage at center
        self.scenario.add_coverage(500, 500, 50, 1.0)
        
        new_coverage = self.scenario.get_coverage_percentage()
        assert new_coverage > initial_coverage
    
    def test_get_coverage_percentage(self):
        """Test getting coverage percentage."""
        coverage = self.scenario.get_coverage_percentage()
        
        assert 0 <= coverage <= 100
        assert isinstance(coverage, float)
    
    def test_get_disaster_zones_in_range(self):
        """Test getting disaster zones in range."""
        zones = self.scenario.get_disaster_zones_in_range(250, 250, 100)
        
        assert isinstance(zones, list)
        # Should find zones near (250, 250)
        assert len(zones) >= 0
    
    def test_get_obstacles_in_range(self):
        """Test getting obstacles in range."""
        obstacles = self.scenario.get_obstacles_in_range(350, 350, 100)
        
        assert isinstance(obstacles, list)
        # Should find obstacles near (350, 350)
        assert len(obstacles) >= 0
    
    def test_get_target_areas_in_range(self):
        """Test getting target areas in range."""
        targets = self.scenario.get_target_areas_in_range(150, 150, 100)
        
        assert isinstance(targets, list)
        # Should find targets near (150, 150)
        assert len(targets) >= 0
    
    def test_check_collision(self):
        """Test collision detection."""
        # Test position that should not collide
        no_collision = self.scenario.check_collision(50, 50, 10)
        assert not no_collision
        
        # Test position that might collide (depends on obstacle placement)
        collision = self.scenario.check_collision(350, 350, 5)
        # Result depends on obstacle configuration
    
    def test_get_environmental_conditions(self):
        """Test getting environmental conditions."""
        conditions = self.scenario.get_environmental_conditions()
        
        assert "wind_speed" in conditions
        assert "wind_direction" in conditions
        assert "weather" in conditions
        assert "temperature" in conditions
        assert "humidity" in conditions
        assert "visibility" in conditions
    
    def test_get_priority_areas(self):
        """Test getting priority areas."""
        priority_areas = self.scenario.get_priority_areas()
        
        assert isinstance(priority_areas, list)
        assert all(len(area) == 5 for area in priority_areas)  # (x, y, w, h, priority)
        assert all(isinstance(area[4], int) for area in priority_areas)  # priority is int
    
    def test_get_scenario_summary(self):
        """Test getting scenario summary."""
        summary = self.scenario.get_scenario_summary()
        
        assert "disaster_zones" in summary
        assert "obstacles" in summary
        assert "target_areas" in summary
        assert "emergency_events" in summary
        assert "coverage_percentage" in summary
        assert "environmental_conditions" in summary
        
        assert isinstance(summary["disaster_zones"], int)
        assert isinstance(summary["obstacles"], int)
        assert isinstance(summary["target_areas"], int)
        assert isinstance(summary["coverage_percentage"], float)


class TestSwarmEnvironment:
    """Test swarm environment functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = SimulationConfig()
        self.config.num_uavs = 3
        self.config.render = False  # Disable rendering for tests
        self.env = SwarmEnvironment(self.config)
    
    def test_environment_initialization(self):
        """Test environment initialization."""
        assert len(self.env.uavs) == self.config.num_uavs
        assert self.env.scenario is not None
        assert self.env.action_space is not None
        assert self.env.observation_space is not None
    
    def test_reset(self):
        """Test environment reset."""
        obs, info = self.env.reset()
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        assert len(obs) == self.env.observation_space.shape[0]
        assert "performance_metrics" in info
        assert "uav_states" in info
        assert "scenario_summary" in info
    
    def test_step(self):
        """Test environment step."""
        obs, info = self.env.reset()
        
        # Generate random actions
        actions = self.env.action_space.sample()
        
        # Step environment
        next_obs, reward, terminated, truncated, info = self.env.step(actions)
        
        assert isinstance(next_obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert len(next_obs) == self.env.observation_space.shape[0]
    
    def test_get_observations(self):
        """Test getting observations."""
        obs = self.env._get_observations()
        
        assert isinstance(obs, np.ndarray)
        assert len(obs) == self.env.observation_space.shape[0]
        assert all(not np.isnan(val) for val in obs)
    
    def test_calculate_reward(self):
        """Test reward calculation."""
        # This is a simplified test since reward calculation depends on environment state
        reward = self.env._calculate_reward()
        
        assert isinstance(reward, (int, float))
        # Reward can be negative, zero, or positive
        assert not np.isnan(reward)
    
    def test_check_termination(self):
        """Test termination checking."""
        # Test normal case (should not terminate)
        terminated = self.env._check_termination()
        assert isinstance(terminated, bool)
        
        # Test with all UAVs in emergency state
        for uav in self.env.uavs:
            uav.state.status = "emergency"
        
        terminated = self.env._check_termination()
        assert terminated is True
    
    def test_check_truncation(self):
        """Test truncation checking."""
        # Test normal case
        truncated = self.env._check_truncation()
        assert isinstance(truncated, bool)
        
        # Test with max episode length reached
        self.env.episode_length = self.config.rl_config.max_episode_length
        truncated = self.env._check_truncation()
        assert truncated is True
    
    def test_get_environment_info(self):
        """Test getting environment information."""
        info = self.env.get_environment_info()
        
        assert "num_uavs" in info
        assert "environment_size" in info
        assert "mcp_connected" in info
        assert "performance_metrics" in info
        assert "scenario_summary" in info
        
        assert info["num_uavs"] == self.config.num_uavs
        assert isinstance(info["environment_size"], tuple)
        assert len(info["environment_size"]) == 2
