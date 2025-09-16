import React from 'react';
import styled from 'styled-components';

const List = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${props => props.theme.spacing.sm};
`;

const UAVItem = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: ${props => props.theme.spacing.md};
  background-color: ${props => props.theme.colors.background};
  border-radius: ${props => props.theme.borderRadius};
  border: 1px solid ${props => props.theme.colors.border};
`;

const UAVInfo = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${props => props.theme.spacing.xs};
`;

const UAVId = styled.span`
  font-weight: 600;
  color: ${props => props.theme.colors.text};
`;

const UAVPosition = styled.span`
  font-size: 0.875rem;
  color: ${props => props.theme.colors.textSecondary};
`;

const UAVStatus = styled.div`
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: ${props => props.theme.spacing.xs};
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

const BatteryLevel = styled.div`
  font-size: 0.875rem;
  color: ${props => props.theme.colors.textSecondary};
`;

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

function UAVList({ uavs = [] }) {
  if (uavs.length === 0) {
    return (
      <div style={{ textAlign: 'center', color: '#64748b', padding: '2rem' }}>
        No UAVs available
      </div>
    );
  }

  return (
    <List>
      {uavs.map((uav) => (
        <UAVItem key={uav.id}>
          <UAVInfo>
            <UAVId>{uav.id}</UAVId>
            <UAVPosition>
              Position: ({uav.position?.x?.toFixed(1) || 0}, {uav.position?.y?.toFixed(1) || 0}, {uav.position?.z?.toFixed(1) || 0})
            </UAVPosition>
          </UAVInfo>
          <UAVStatus>
            <StatusBadge status={uav.status}>
              {uav.status.replace('_', ' ')}
            </StatusBadge>
            <BatteryLevel>
              Battery: {uav.battery?.toFixed(1) || 0}%
            </BatteryLevel>
          </UAVStatus>
        </UAVItem>
      ))}
    </List>
  );
}

export default UAVList;
