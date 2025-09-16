import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { useSocket } from '../services/SocketService';

const AgentContainer = styled.div`
  max-width: 1200px;
  margin: 0 auto;
`;

const Title = styled.h1`
  font-size: 2rem;
  font-weight: bold;
  color: ${props => props.theme.colors.text};
  margin-bottom: ${props => props.theme.spacing.xl};
`;

const AgentGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
  gap: ${props => props.theme.spacing.lg};
`;

const AgentCard = styled.div`
  background-color: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.borderRadius};
  padding: ${props => props.theme.spacing.lg};
  box-shadow: ${props => props.theme.shadows.sm};
  border-left: 4px solid ${props => getAgentColor(props.status)};
`;

const AgentHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: ${props => props.theme.spacing.md};
`;

const AgentId = styled.h3`
  font-size: 1.25rem;
  font-weight: 600;
  color: ${props => props.theme.colors.text};
`;

const StatusBadge = styled.span`
  padding: ${props => props.theme.spacing.xs} ${props => props.theme.spacing.sm};
  border-radius: ${props => props.theme.borderRadius};
  font-size: 0.75rem;
  font-weight: 500;
  text-transform: uppercase;
  background-color: ${props => getStatusColor(props.status).background};
  color: ${props => getStatusColor(props.status).text};
`;

const AgentInfo = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: ${props => props.theme.spacing.md};
  margin-bottom: ${props => props.theme.spacing.md};
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
  font-size: 1rem;
  color: ${props => props.theme.colors.text};
  font-weight: 600;
`;

const ProgressBar = styled.div`
  width: 100%;
  height: 8px;
  background-color: ${props => props.theme.colors.border};
  border-radius: 4px;
  overflow: hidden;
  margin-top: ${props => props.theme.spacing.xs};
`;

const ProgressFill = styled.div`
  height: 100%;
  background-color: ${props => props.color};
  width: ${props => props.percentage}%;
  transition: width 0.3s ease;
`;

const getAgentColor = (status) => {
  const colors = {
    active: '#10b981',
    low_battery: '#f59e0b',
    emergency: '#ef4444',
    charging: '#3b82f6',
    maintenance: '#6b7280'
  };
  return colors[status] || colors.active;
};

const getStatusColor = (status) => {
  const colors = {
    active: { background: '#dcfce7', text: '#166534' },
    low_battery: { background: '#fef3c7', text: '#92400e' },
    emergency: { background: '#fee2e2', text: '#991b1b' },
    charging: { background: '#dbeafe', text: '#1e40af' },
    maintenance: { background: '#f3f4f6', text: '#374151' }
  };
  return colors[status] || colors.active;
};

const getProgressColor = (value, type) => {
  if (type === 'battery') {
    if (value > 50) return '#10b981';
    if (value > 20) return '#f59e0b';
    return '#ef4444';
  }
  return '#2563eb';
};

function AgentView() {
  const socket = useSocket();
  const [agentsData, setAgentsData] = useState({});
  const [simulationData, setSimulationData] = useState({
    uavs: [],
    performance: {}
  });

  useEffect(() => {
    if (!socket) return;

    const handleAgentData = (data) => {
      setAgentsData(data);
    };

    const handleSimulationUpdate = (data) => {
      setSimulationData(data);
    };

    socket.on('agent_data', handleAgentData);
    socket.on('simulation_update', handleSimulationUpdate);

    socket.emit('request_agent_data');
    socket.emit('request_simulation_data');

    return () => {
      socket.off('agent_data', handleAgentData);
      socket.off('simulation_update', handleSimulationUpdate);
    };
  }, [socket]);

  const agents = Object.values(agentsData);
  const uavs = simulationData.uavs || [];

  return (
    <AgentContainer>
      <Title>Agent Status</Title>
      
      <AgentGrid>
        {uavs.map((uav, index) => (
          <AgentCard key={uav.id || index} status={uav.status}>
            <AgentHeader>
              <AgentId>{uav.id || `UAV-${index}`}</AgentId>
              <StatusBadge status={uav.status}>
                {uav.status?.replace('_', ' ') || 'Unknown'}
              </StatusBadge>
            </AgentHeader>
            
            <AgentInfo>
              <InfoItem>
                <InfoLabel>Position</InfoLabel>
                <InfoValue>
                  ({uav.position?.x?.toFixed(1) || 0}, {uav.position?.y?.toFixed(1) || 0}, {uav.position?.z?.toFixed(1) || 0})
                </InfoValue>
              </InfoItem>
              
              <InfoItem>
                <InfoLabel>Communication Range</InfoLabel>
                <InfoValue>{uav.communication_range || 50}m</InfoValue>
              </InfoItem>
              
              <InfoItem>
                <InfoLabel>Sensor Range</InfoLabel>
                <InfoValue>{uav.sensor_range || 30}m</InfoValue>
              </InfoItem>
              
              <InfoItem>
                <InfoLabel>Payload</InfoLabel>
                <InfoValue>{uav.payload || 0}kg</InfoValue>
              </InfoItem>
            </AgentInfo>
            
            <InfoItem>
              <InfoLabel>Battery Level</InfoLabel>
              <InfoValue>{uav.battery?.toFixed(1) || 0}%</InfoValue>
              <ProgressBar>
                <ProgressFill 
                  percentage={uav.battery || 0} 
                  color={getProgressColor(uav.battery || 0, 'battery')}
                />
              </ProgressBar>
            </InfoItem>
          </AgentCard>
        ))}
        
        {agents.map((agent, index) => (
          <AgentCard key={agent.agent_id || index} status="active">
            <AgentHeader>
              <AgentId>{agent.agent_id || `Agent-${index}`}</AgentId>
              <StatusBadge status="active">Active</StatusBadge>
            </AgentHeader>
            
            <AgentInfo>
              <InfoItem>
                <InfoLabel>Agent Type</InfoLabel>
                <InfoValue>{agent.agent_type || 'Unknown'}</InfoValue>
              </InfoItem>
              
              <InfoItem>
                <InfoLabel>Last Update</InfoLabel>
                <InfoValue>
                  {agent.last_update 
                    ? new Date(agent.last_update).toLocaleTimeString()
                    : 'Never'
                  }
                </InfoValue>
              </InfoItem>
              
              <InfoItem>
                <InfoLabel>Capabilities</InfoLabel>
                <InfoValue>
                  {agent.capabilities?.join(', ') || 'None'}
                </InfoValue>
              </InfoItem>
              
              <InfoItem>
                <InfoLabel>Context Requirements</InfoLabel>
                <InfoValue>
                  {agent.context_requirements?.join(', ') || 'None'}
                </InfoValue>
              </InfoItem>
            </AgentInfo>
          </AgentCard>
        ))}
        
        {uavs.length === 0 && agents.length === 0 && (
          <div style={{ 
            gridColumn: '1 / -1', 
            textAlign: 'center', 
            color: '#64748b', 
            padding: '2rem' 
          }}>
            No agents or UAVs available
          </div>
        )}
      </AgentGrid>
    </AgentContainer>
  );
}

export default AgentView;
