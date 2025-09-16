import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import styled from 'styled-components';

const ChartContainer = styled.div`
  height: 400px;
  width: 100%;
`;

const ChartTitle = styled.h3`
  font-size: 1rem;
  font-weight: 600;
  color: ${props => props.theme.colors.text};
  margin-bottom: ${props => props.theme.spacing.md};
`;

const NoDataMessage = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  height: 400px;
  color: ${props => props.theme.colors.textSecondary};
  font-style: italic;
`;

function PerformanceChart({ coverageData = [], batteryData = [] }) {
  if (coverageData.length === 0 && batteryData.length === 0) {
    return (
      <ChartContainer>
        <ChartTitle>Performance Metrics</ChartTitle>
        <NoDataMessage>No performance data available</NoDataMessage>
      </ChartContainer>
    );
  }

  // Combine data for the chart
  const maxLength = Math.max(coverageData.length, batteryData.length);
  const chartData = [];

  for (let i = 0; i < maxLength; i++) {
    chartData.push({
      time: i,
      coverage: coverageData[i]?.value || 0,
      battery: batteryData[i]?.value || 0,
      timestamp: coverageData[i]?.timestamp || batteryData[i]?.timestamp
    });
  }

  return (
    <ChartContainer>
      <ChartTitle>Performance Metrics Over Time</ChartTitle>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="time" 
            tick={{ fontSize: 12 }}
            tickFormatter={(value) => `T${value}`}
          />
          <YAxis 
            domain={[0, 100]}
            tick={{ fontSize: 12 }}
            tickFormatter={(value) => `${value}%`}
          />
          <Tooltip 
            formatter={(value, name) => [`${value.toFixed(1)}%`, name === 'coverage' ? 'Coverage' : 'Battery']}
            labelFormatter={(label) => `Time: ${label}`}
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="coverage" 
            stroke="#2563eb" 
            strokeWidth={2}
            dot={{ fill: '#2563eb', strokeWidth: 2, r: 4 }}
            name="Coverage"
          />
          <Line 
            type="monotone" 
            dataKey="battery" 
            stroke="#f59e0b" 
            strokeWidth={2}
            dot={{ fill: '#f59e0b', strokeWidth: 2, r: 4 }}
            name="Battery"
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartContainer>
  );
}

export default PerformanceChart;
