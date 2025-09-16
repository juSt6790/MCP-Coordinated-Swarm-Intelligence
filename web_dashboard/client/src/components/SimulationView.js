import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { useSocket } from '../services/SocketService';

const SimulationContainer = styled.div`
  max-width: 1200px;
  margin: 0 auto;
`;

const Title = styled.h1`
  font-size: 2rem;
  font-weight: bold;
  color: ${props => props.theme.colors.text};
  margin-bottom: ${props => props.theme.spacing.xl};
`;

const SimulationCanvas = styled.div`
  background-color: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.borderRadius};
  padding: ${props => props.theme.spacing.lg};
  box-shadow: ${props => props.theme.shadows.sm};
  margin-bottom: ${props => props.theme.spacing.lg};
  height: 600px;
  position: relative;
  overflow: hidden;
`;

const UAV = styled.div`
  position: absolute;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background-color: ${props => getUAVColor(props.status)};
  border: 2px solid ${props => props.theme.colors.surface};
  transform: translate(-50%, -50%);
  transition: all 0.3s ease;
  cursor: pointer;
  
  &:hover {
    transform: translate(-50%, -50%) scale(1.5);
    z-index: 10;
  }
`;

const UAVTooltip = styled.div`
  position: absolute;
  background-color: ${props => props.theme.colors.text};
  color: ${props => props.theme.colors.surface};
  padding: ${props => props.theme.spacing.sm};
  border-radius: ${props => props.theme.borderRadius};
  font-size: 0.875rem;
  white-space: nowrap;
  z-index: 20;
  pointer-events: none;
  opacity: ${props => props.visible ? 1 : 0};
  transition: opacity 0.2s ease;
`;

const getUAVColor = (status) => {
  const colors = {
    active: '#10b981',
    low_battery: '#f59e0b',
    emergency: '#ef4444',
    charging: '#3b82f6',
    maintenance: '#6b7280'
  };
  return colors[status] || colors.active;
};

const Controls = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.md};
  margin-bottom: ${props => props.theme.spacing.lg};
`;

const Button = styled.button`
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
  border: none;
  border-radius: ${props => props.theme.borderRadius};
  background-color: ${props => props.theme.colors.primary};
  color: ${props => props.theme.colors.surface};
  font-weight: 500;
  cursor: pointer;
  transition: background-color 0.2s ease;

  &:hover {
    background-color: ${props => props.theme.colors.primary}dd;
  }

  &:disabled {
    background-color: ${props => props.theme.colors.secondary};
    cursor: not-allowed;
  }
`;

const InfoPanel = styled.div`
  background-color: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.borderRadius};
  padding: ${props => props.theme.spacing.lg};
  box-shadow: ${props => props.theme.shadows.sm};
`;

const InfoGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: ${props => props.theme.spacing.md};
`;

const InfoItem = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${props => props.theme.spacing.xs};
`;

const InfoLabel = styled.span`
  font-size: 0.875rem;
  color: ${props => props.theme.colors.textSecondary};
  font-weight: 500;
`;

const InfoValue = styled.span`
  font-size: 1.125rem;
  color: ${props => props.theme.colors.text};
  font-weight: 600;
`;

function SimulationView() {
  const socket = useSocket();
  const [simulationData, setSimulationData] = useState({
    uavs: [],
    scenario: {},
    performance: {},
    timestamp: null
  });
  const [hoveredUAV, setHoveredUAV] = useState(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    if (!socket) return;

    const handleSimulationUpdate = (data) => {
      setSimulationData(data);
    };

    socket.on('simulation_update', handleSimulationUpdate);
    socket.emit('request_simulation_data');

    return () => {
      socket.off('simulation_update', handleSimulationUpdate);
    };
  }, [socket]);

  const handleMouseMove = (e) => {
    setMousePosition({ x: e.clientX, y: e.clientY });
  };

  const handleUAVHover = (uav) => {
    setHoveredUAV(uav);
  };

  const handleUAVLeave = () => {
    setHoveredUAV(null);
  };

  const startSimulation = () => {
    // This would trigger simulation start in a real implementation
    console.log('Starting simulation...');
  };

  const stopSimulation = () => {
    // This would trigger simulation stop in a real implementation
    console.log('Stopping simulation...');
  };

  const resetSimulation = () => {
    // This would trigger simulation reset in a real implementation
    console.log('Resetting simulation...');
  };

  return (
    <SimulationContainer>
      <Title>Simulation View</Title>
      
      <Controls>
        <Button onClick={startSimulation}>Start Simulation</Button>
        <Button onClick={stopSimulation}>Stop Simulation</Button>
        <Button onClick={resetSimulation}>Reset Simulation</Button>
      </Controls>

      <SimulationCanvas onMouseMove={handleMouseMove}>
        {simulationData.uavs.map((uav) => (
          <UAV
            key={uav.id}
            status={uav.status}
            style={{
              left: `${(uav.position?.x / 1000) * 100}%`,
              top: `${(uav.position?.y / 1000) * 100}%`
            }}
            onMouseEnter={() => handleUAVHover(uav)}
            onMouseLeave={handleUAVLeave}
          />
        ))}
        
        {hoveredUAV && (
          <UAVTooltip
            visible={true}
            style={{
              left: mousePosition.x + 10,
              top: mousePosition.y - 10
            }}
          >
            <div><strong>{hoveredUAV.id}</strong></div>
            <div>Status: {hoveredUAV.status}</div>
            <div>Battery: {hoveredUAV.battery?.toFixed(1)}%</div>
            <div>
              Position: ({hoveredUAV.position?.x?.toFixed(1)}, {hoveredUAV.position?.y?.toFixed(1)}, {hoveredUAV.position?.z?.toFixed(1)})
            </div>
          </UAVTooltip>
        )}
      </SimulationCanvas>

      <InfoPanel>
        <h3 style={{ marginBottom: '1rem', color: '#1e293b' }}>Simulation Information</h3>
        <InfoGrid>
          <InfoItem>
            <InfoLabel>Total UAVs</InfoLabel>
            <InfoValue>{simulationData.uavs.length}</InfoValue>
          </InfoItem>
          <InfoItem>
            <InfoLabel>Active UAVs</InfoLabel>
            <InfoValue>
              {simulationData.uavs.filter(uav => uav.status === 'active').length}
            </InfoValue>
          </InfoItem>
          <InfoItem>
            <InfoLabel>Coverage</InfoLabel>
            <InfoValue>
              {simulationData.performance?.coverage_percentage?.toFixed(1) || 0}%
            </InfoValue>
          </InfoItem>
          <InfoItem>
            <InfoLabel>Last Update</InfoLabel>
            <InfoValue>
              {simulationData.timestamp 
                ? new Date(simulationData.timestamp).toLocaleTimeString()
                : 'Never'
              }
            </InfoValue>
          </InfoItem>
        </InfoGrid>
      </InfoPanel>
    </SimulationContainer>
  );
}

export default SimulationView;
