import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { useSocket } from '../services/SocketService';
import MetricCard from './MetricCard';
import UAVList from './UAVList';
import CoverageChart from './CoverageChart';
import PerformanceChart from './PerformanceChart';

const DashboardContainer = styled.div`
  max-width: 1200px;
  margin: 0 auto;
`;

const Title = styled.h1`
  font-size: 2rem;
  font-weight: bold;
  color: ${props => props.theme.colors.text};
  margin-bottom: ${props => props.theme.spacing.xl};
`;

const Grid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: ${props => props.theme.spacing.lg};
  margin-bottom: ${props => props.theme.spacing.xl};
`;

const Section = styled.section`
  background-color: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.borderRadius};
  padding: ${props => props.theme.spacing.lg};
  box-shadow: ${props => props.theme.shadows.sm};
`;

const SectionTitle = styled.h2`
  font-size: 1.25rem;
  font-weight: 600;
  color: ${props => props.theme.colors.text};
  margin-bottom: ${props => props.theme.spacing.md};
`;

const ChartsGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: ${props => props.theme.spacing.lg};
  margin-top: ${props => props.theme.spacing.lg};

  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

function Dashboard() {
  const socket = useSocket();
  const [simulationData, setSimulationData] = useState({
    uavs: [],
    scenario: {},
    performance: {},
    timestamp: null
  });
  const [performanceMetrics, setPerformanceMetrics] = useState({
    coverage: [],
    battery: [],
    communication: [],
    rewards: []
  });

  useEffect(() => {
    if (!socket) return;

    const handleSimulationUpdate = (data) => {
      setSimulationData(data);
    };

    const handlePerformanceData = (data) => {
      setPerformanceMetrics(data);
    };

    socket.on('simulation_update', handleSimulationUpdate);
    socket.on('performance_data', handlePerformanceData);

    // Request initial data
    socket.emit('request_simulation_data');
    socket.emit('request_performance_data');

    return () => {
      socket.off('simulation_update', handleSimulationUpdate);
      socket.off('performance_data', handlePerformanceData);
    };
  }, [socket]);

  const calculateMetrics = () => {
    const uavs = simulationData.uavs || [];
    const performance = simulationData.performance || {};
    
    return {
      totalUAVs: uavs.length,
      activeUAVs: uavs.filter(uav => uav.status === 'active').length,
      averageBattery: uavs.length > 0 
        ? (uavs.reduce((sum, uav) => sum + uav.battery, 0) / uavs.length).toFixed(1)
        : 0,
      coveragePercentage: performance.coverage_percentage || 0,
      emergencyUAVs: uavs.filter(uav => uav.status === 'emergency').length,
      lowBatteryUAVs: uavs.filter(uav => uav.status === 'low_battery').length
    };
  };

  const metrics = calculateMetrics();

  return (
    <DashboardContainer>
      <Title>Swarm Intelligence Dashboard</Title>
      
      <Grid>
        <MetricCard
          title="Total UAVs"
          value={metrics.totalUAVs}
          icon="🚁"
          color="primary"
        />
        <MetricCard
          title="Active UAVs"
          value={metrics.activeUAVs}
          icon="✅"
          color="success"
        />
        <MetricCard
          title="Average Battery"
          value={`${metrics.averageBattery}%`}
          icon="🔋"
          color="warning"
        />
        <MetricCard
          title="Coverage"
          value={`${metrics.coveragePercentage.toFixed(1)}%`}
          icon="📊"
          color="primary"
        />
        <MetricCard
          title="Emergency UAVs"
          value={metrics.emergencyUAVs}
          icon="⚠️"
          color="error"
        />
        <MetricCard
          title="Low Battery"
          value={metrics.lowBatteryUAVs}
          icon="🔋"
          color="warning"
        />
      </Grid>

      <ChartsGrid>
        <Section>
          <SectionTitle>UAV Status</SectionTitle>
          <UAVList uavs={simulationData.uavs} />
        </Section>
        
        <Section>
          <SectionTitle>Coverage Progress</SectionTitle>
          <CoverageChart data={performanceMetrics.coverage} />
        </Section>
      </ChartsGrid>

      <Section>
        <SectionTitle>Performance Metrics</SectionTitle>
        <PerformanceChart 
          coverageData={performanceMetrics.coverage}
          batteryData={performanceMetrics.battery}
        />
      </Section>
    </DashboardContainer>
  );
}

export default Dashboard;
