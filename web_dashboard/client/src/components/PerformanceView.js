import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { useSocket } from '../services/SocketService';
import PerformanceChart from './PerformanceChart';
import MetricCard from './MetricCard';

const PerformanceContainer = styled.div`
  max-width: 1200px;
  margin: 0 auto;
`;

const Title = styled.h1`
  font-size: 2rem;
  font-weight: bold;
  color: ${props => props.theme.colors.text};
  margin-bottom: ${props => props.theme.spacing.xl};
`;

const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: ${props => props.theme.spacing.lg};
  margin-bottom: ${props => props.theme.spacing.xl};
`;

const ChartSection = styled.section`
  background-color: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.borderRadius};
  padding: ${props => props.theme.spacing.lg};
  box-shadow: ${props => props.theme.shadows.sm};
  margin-bottom: ${props => props.theme.spacing.lg};
`;

const SectionTitle = styled.h2`
  font-size: 1.25rem;
  font-weight: 600;
  color: ${props => props.theme.colors.text};
  margin-bottom: ${props => props.theme.spacing.md};
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: ${props => props.theme.spacing.md};
  margin-top: ${props => props.theme.spacing.lg};
`;

const StatItem = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${props => props.theme.spacing.xs};
`;

const StatLabel = styled.span`
  font-size: 0.875rem;
  color: ${props => props.theme.colors.textSecondary};
  font-weight: 500;
`;

const StatValue = styled.span`
  font-size: 1.5rem;
  color: ${props => props.theme.colors.text};
  font-weight: bold;
`;

function PerformanceView() {
  const socket = useSocket();
  const [performanceMetrics, setPerformanceMetrics] = useState({
    coverage: [],
    battery: [],
    communication: [],
    rewards: []
  });
  const [simulationData, setSimulationData] = useState({
    uavs: [],
    performance: {}
  });

  useEffect(() => {
    if (!socket) return;

    const handlePerformanceData = (data) => {
      setPerformanceMetrics(data);
    };

    const handleSimulationUpdate = (data) => {
      setSimulationData(data);
    };

    socket.on('performance_data', handlePerformanceData);
    socket.on('simulation_update', handleSimulationUpdate);

    socket.emit('request_performance_data');
    socket.emit('request_simulation_data');

    return () => {
      socket.off('performance_data', handlePerformanceData);
      socket.off('simulation_update', handleSimulationUpdate);
    };
  }, [socket]);

  const calculateStats = () => {
    const coverage = performanceMetrics.coverage || [];
    const battery = performanceMetrics.battery || [];
    const communication = performanceMetrics.communication || [];
    const rewards = performanceMetrics.rewards || [];

    return {
      avgCoverage: coverage.length > 0 
        ? (coverage.reduce((sum, item) => sum + item.value, 0) / coverage.length).toFixed(1)
        : 0,
      maxCoverage: coverage.length > 0 
        ? Math.max(...coverage.map(item => item.value)).toFixed(1)
        : 0,
      avgBattery: battery.length > 0 
        ? (battery.reduce((sum, item) => sum + item.value, 0) / battery.length).toFixed(1)
        : 0,
      minBattery: battery.length > 0 
        ? Math.min(...battery.map(item => item.value)).toFixed(1)
        : 0,
      avgCommunication: communication.length > 0 
        ? (communication.reduce((sum, item) => sum + item.value, 0) / communication.length).toFixed(1)
        : 0,
      avgReward: rewards.length > 0 
        ? (rewards.reduce((sum, item) => sum + item.value, 0) / rewards.length).toFixed(2)
        : 0,
      totalDataPoints: Math.max(coverage.length, battery.length, communication.length, rewards.length)
    };
  };

  const stats = calculateStats();

  return (
    <PerformanceContainer>
      <Title>Performance Analysis</Title>
      
      <MetricsGrid>
        <MetricCard
          title="Average Coverage"
          value={`${stats.avgCoverage}%`}
          icon="📊"
          color="primary"
        />
        <MetricCard
          title="Max Coverage"
          value={`${stats.maxCoverage}%`}
          icon="🎯"
          color="success"
        />
        <MetricCard
          title="Average Battery"
          value={`${stats.avgBattery}%`}
          icon="🔋"
          color="warning"
        />
        <MetricCard
          title="Min Battery"
          value={`${stats.minBattery}%`}
          icon="⚠️"
          color="error"
        />
        <MetricCard
          title="Communication"
          value={`${stats.avgCommunication}%`}
          icon="📡"
          color="primary"
        />
        <MetricCard
          title="Average Reward"
          value={stats.avgReward}
          icon="🏆"
          color="success"
        />
      </MetricsGrid>

      <ChartSection>
        <SectionTitle>Performance Trends</SectionTitle>
        <PerformanceChart 
          coverageData={performanceMetrics.coverage}
          batteryData={performanceMetrics.battery}
        />
      </ChartSection>

      <ChartSection>
        <SectionTitle>Detailed Statistics</SectionTitle>
        <StatsGrid>
          <StatItem>
            <StatLabel>Data Points Collected</StatLabel>
            <StatValue>{stats.totalDataPoints}</StatValue>
          </StatItem>
          <StatItem>
            <StatLabel>Current Coverage</StatLabel>
            <StatValue>{simulationData.performance?.coverage_percentage?.toFixed(1) || 0}%</StatValue>
          </StatItem>
          <StatItem>
            <StatLabel>Active UAVs</StatLabel>
            <StatValue>
              {simulationData.uavs?.filter(uav => uav.status === 'active').length || 0}
            </StatValue>
          </StatItem>
          <StatItem>
            <StatLabel>Total UAVs</StatLabel>
            <StatValue>{simulationData.uavs?.length || 0}</StatValue>
          </StatItem>
        </StatsGrid>
      </ChartSection>
    </PerformanceContainer>
  );
}

export default PerformanceView;
