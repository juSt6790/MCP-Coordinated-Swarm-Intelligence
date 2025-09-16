import React from 'react';
import styled from 'styled-components';

const Card = styled.div`
  background-color: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.borderRadius};
  padding: ${props => props.theme.spacing.lg};
  box-shadow: ${props => props.theme.shadows.sm};
  border-left: 4px solid ${props => getColor(props.color)};
  transition: transform 0.2s ease, box-shadow 0.2s ease;

  &:hover {
    transform: translateY(-2px);
    box-shadow: ${props => props.theme.shadows.md};
  }
`;

const CardHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: ${props => props.theme.spacing.sm};
`;

const Title = styled.h3`
  font-size: 0.875rem;
  font-weight: 500;
  color: ${props => props.theme.colors.textSecondary};
  text-transform: uppercase;
  letter-spacing: 0.05em;
`;

const Icon = styled.span`
  font-size: 1.5rem;
`;

const Value = styled.div`
  font-size: 2rem;
  font-weight: bold;
  color: ${props => props.theme.colors.text};
`;

const getColor = (color) => {
  const colors = {
    primary: '#2563eb',
    secondary: '#64748b',
    success: '#10b981',
    warning: '#f59e0b',
    error: '#ef4444'
  };
  return colors[color] || colors.primary;
};

function MetricCard({ title, value, icon, color = 'primary' }) {
  return (
    <Card color={color}>
      <CardHeader>
        <Title>{title}</Title>
        <Icon>{icon}</Icon>
      </CardHeader>
      <Value>{value}</Value>
    </Card>
  );
}

export default MetricCard;
