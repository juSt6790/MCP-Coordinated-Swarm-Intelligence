import { createContext, useContext } from 'react';

export const SocketContext = createContext(null);

export const useSocket = () => {
  // May be null initially until connection is established; components should guard
  return useContext(SocketContext);
};
