"""Tests for MCP server functionality."""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch

from mcp_server.server import MCPServer
from mcp_server.context_manager import ContextManager
from mcp_server.message_protocol import MessageProtocol
from config.mcp_config import MCPConfig, ContextMessage


class TestMCPConfig:
    """Test MCP configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MCPConfig()
        assert config.host == "localhost"
        assert config.port == 8765
        assert config.max_connections == 50
        assert config.context_update_frequency == 1.0
    
    def test_config_from_yaml(self, tmp_path):
        """Test loading configuration from YAML file."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
host: "test_host"
port: 9999
max_connections: 100
context_update_frequency: 2.0
""")
        
        config = MCPConfig.from_yaml(str(config_file))
        assert config.host == "test_host"
        assert config.port == 9999
        assert config.max_connections == 100
        assert config.context_update_frequency == 2.0


class TestContextMessage:
    """Test context message functionality."""
    
    def test_message_creation(self):
        """Test creating a context message."""
        message = ContextMessage(
            message_type="update",
            sender_id="test_agent",
            timestamp=1234567890.0,
            context_type="position",
            data={"x": 100, "y": 200},
            priority=2
        )
        
        assert message.message_type == "update"
        assert message.sender_id == "test_agent"
        assert message.context_type == "position"
        assert message.data == {"x": 100, "y": 200}
        assert message.priority == 2
    
    def test_message_serialization(self):
        """Test message serialization to dictionary."""
        message = ContextMessage(
            message_type="update",
            sender_id="test_agent",
            timestamp=1234567890.0,
            context_type="position",
            data={"x": 100, "y": 200},
            priority=2
        )
        
        data = message.to_dict()
        assert data["message_type"] == "update"
        assert data["sender_id"] == "test_agent"
        assert data["context_type"] == "position"
        assert data["data"] == {"x": 100, "y": 200}
        assert data["priority"] == 2
    
    def test_message_deserialization(self):
        """Test message deserialization from dictionary."""
        data = {
            "message_type": "update",
            "sender_id": "test_agent",
            "timestamp": 1234567890.0,
            "context_type": "position",
            "data": {"x": 100, "y": 200},
            "priority": 2
        }
        
        message = ContextMessage.from_dict(data)
        assert message.message_type == "update"
        assert message.sender_id == "test_agent"
        assert message.context_type == "position"
        assert message.data == {"x": 100, "y": 200}
        assert message.priority == 2


class TestMessageProtocol:
    """Test message protocol functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.protocol = MessageProtocol()
    
    def test_register_handler(self):
        """Test registering message handlers."""
        handler = Mock()
        self.protocol.register_handler("test_message", handler)
        assert "test_message" in self.protocol.message_handlers
        assert self.protocol.message_handlers["test_message"] == handler
    
    def test_serialize_message(self):
        """Test message serialization."""
        message = ContextMessage(
            message_type="test",
            sender_id="test_agent",
            timestamp=1234567890.0,
            context_type="test",
            data={"test": "data"}
        )
        
        serialized = self.protocol.serialize_message(message)
        assert isinstance(serialized, str)
        
        # Test that it can be deserialized
        deserialized = self.protocol.deserialize_message(serialized)
        assert deserialized.message_type == "test"
        assert deserialized.sender_id == "test_agent"
    
    def test_deserialize_invalid_message(self):
        """Test deserializing invalid message."""
        result = self.protocol.deserialize_message("invalid json")
        assert result is None
    
    def test_register_client(self):
        """Test client registration."""
        websocket = Mock()
        client_info = {"type": "test_client"}
        
        self.protocol.register_client("test_client", websocket, client_info)
        assert "test_client" in self.protocol.registered_clients
        assert self.protocol.registered_clients["test_client"]["websocket"] == websocket
        assert self.protocol.registered_clients["test_client"]["info"] == client_info


class TestContextManager:
    """Test context manager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        config = MCPConfig()
        self.context_manager = ContextManager(config)
    
    def test_initialization(self):
        """Test context manager initialization."""
        assert self.context_manager.config is not None
        assert self.context_manager.context_history is not None
        assert self.context_manager.coverage_grid is not None
    
    def test_update_client_data(self):
        """Test updating client data."""
        message = ContextMessage(
            message_type="update",
            sender_id="test_client",
            timestamp=1234567890.0,
            context_type="position",
            data={"x": 100, "y": 200, "sensor_range": 30}
        )
        
        self.context_manager.update_client_data("test_client", message)
        assert "test_client" in self.context_manager.client_data
        assert self.context_manager.client_data["test_client"]["context_type"] == "position"
    
    def test_aggregate_context(self):
        """Test context aggregation."""
        # Add some test data
        message = ContextMessage(
            message_type="update",
            sender_id="test_client",
            timestamp=1234567890.0,
            context_type="position",
            data={"x": 100, "y": 200, "sensor_range": 30}
        )
        self.context_manager.update_client_data("test_client", message)
        
        context = self.context_manager.aggregate_context()
        assert context is not None
        assert "coverage_map" in context.to_dict()
        assert "battery_status" in context.to_dict()
        assert "communication_network" in context.to_dict()
    
    def test_get_coverage_statistics(self):
        """Test getting coverage statistics."""
        stats = self.context_manager.get_coverage_statistics()
        assert "coverage_percentage" in stats
        assert "uncovered_areas" in stats
        assert "last_updated" in stats


@pytest.mark.asyncio
class TestMCPServer:
    """Test MCP server functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = MCPConfig()
        self.server = MCPServer(self.config)
    
    async def test_handle_update_message(self):
        """Test handling update messages."""
        message = ContextMessage(
            message_type="update",
            sender_id="test_client",
            timestamp=1234567890.0,
            context_type="position",
            data={"x": 100, "y": 200}
        )
        
        response = await self.server._handle_update("test_client", message)
        assert response is not None
        
        # Parse response
        response_data = json.loads(response)
        assert response_data["message_type"] == "acknowledgment"
        assert response_data["data"]["client_id"] == "test_client"
    
    async def test_handle_query_message(self):
        """Test handling query messages."""
        message = ContextMessage(
            message_type="query",
            sender_id="test_client",
            timestamp=1234567890.0,
            context_type="swarm_context",
            data={"query_type": "full_context"}
        )
        
        response = await self.server._handle_query("test_client", message)
        assert response is not None
        
        # Parse response
        response_data = json.loads(response)
        assert response_data["message_type"] == "query_response"
        assert "context" in response_data["data"]
    
    async def test_handle_heartbeat_message(self):
        """Test handling heartbeat messages."""
        message = ContextMessage(
            message_type="heartbeat",
            sender_id="test_client",
            timestamp=1234567890.0,
            context_type="heartbeat",
            data={}
        )
        
        response = await self.server._handle_heartbeat("test_client", message)
        assert response is not None
        
        # Parse response
        response_data = json.loads(response)
        assert response_data["message_type"] == "heartbeat_response"
        assert response_data["data"]["status"] == "alive"
    
    async def test_handle_register_message(self):
        """Test handling registration messages."""
        message = ContextMessage(
            message_type="register",
            sender_id="test_client",
            timestamp=1234567890.0,
            context_type="registration",
            data={"client_info": {"type": "test"}}
        )
        
        response = await self.server._handle_register("test_client", message)
        assert response is not None
        
        # Parse response
        response_data = json.loads(response)
        assert response_data["message_type"] == "registration_confirmed"
        assert response_data["data"]["client_id"] == "test_client"
