const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const path = require('path');
const fs = require('fs');
require('dotenv').config();

const WebSocketHandler = require('./websocket_handler');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

const PORT = process.env.PORT || 3001;

// Middleware
app.use(helmet());
app.use(cors());
app.use(morgan('combined'));
app.use(express.json());
const clientBuildPath = path.join(__dirname, '../client/build');
const hasClientBuild = fs.existsSync(path.join(clientBuildPath, 'index.html'));

if (hasClientBuild) {
  app.use(express.static(clientBuildPath));
}

// WebSocket handler
const wsHandler = new WebSocketHandler(io);

// API Routes
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    timestamp: new Date().toISOString(),
    mcp_connected: wsHandler.isMCPConnected()
  });
});

app.get('/api/simulation/status', (req, res) => {
  res.json(wsHandler.getSimulationStatus());
});

app.get('/api/agents', (req, res) => {
  res.json(wsHandler.getAgentsInfo());
});

app.get('/api/performance', (req, res) => {
  res.json(wsHandler.getPerformanceMetrics());
});

// Serve React app if built; otherwise provide guidance
if (hasClientBuild) {
  app.get('*', (req, res) => {
    res.sendFile(path.join(clientBuildPath, 'index.html'));
  });
} else {
  app.get('/', (req, res) => {
    res.status(200).send({
      message: 'Dashboard UI not built yet. Run: cd web_dashboard/client && npm install && npm run build',
      api_endpoints: ['/api/health', '/api/simulation/status', '/api/agents', '/api/performance']
    });
  });
}

// WebSocket connection handling
io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);
  
  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
  
  socket.on('request_simulation_data', () => {
    wsHandler.sendSimulationData(socket);
  });
  
  socket.on('request_agent_data', () => {
    wsHandler.sendAgentData(socket);
  });
  
  socket.on('request_performance_data', () => {
    wsHandler.sendPerformanceData(socket);
  });
});

// Start server
server.listen(PORT, () => {
  console.log(`Dashboard server running on port ${PORT}`);
  console.log(`WebSocket server ready for connections`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully');
  server.close(() => {
    console.log('Process terminated');
  });
});

process.on('SIGINT', () => {
  console.log('SIGINT received, shutting down gracefully');
  server.close(() => {
    console.log('Process terminated');
  });
});
