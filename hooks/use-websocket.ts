'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';
import { LiveEvent } from '@/lib/types/api';

// Socket.io uses HTTP/HTTPS URLs, not ws:// URLs
// It will automatically upgrade to WebSocket
// Connect through API Gateway for proper routing and authentication
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3000';
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || API_URL;

interface UseWebSocketOptions {
  autoConnect?: boolean;
  organizationId?: string;
  apiKey?: string;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Error) => void;
}

interface UseWebSocketReturn {
  socket: Socket | null;
  isConnected: boolean;
  events: LiveEvent[];
  connect: () => void;
  disconnect: () => void;
  subscribe: (organizationId: string) => void;
  unsubscribe: (organizationId: string) => void;
  clearEvents: () => void;
}

// Simple logger for client-side (console in development)
const logger = {
  debug: (...args: any[]) => {
    if (process.env.NODE_ENV === 'development') {
      console.debug('[WebSocket]', ...args);
    }
  },
  error: (...args: any[]) => {
    console.error('[WebSocket]', ...args);
  },
};

export function useWebSocket(options: UseWebSocketOptions = {}): UseWebSocketReturn {
  const {
    autoConnect = true,
    organizationId,
    apiKey,
    onConnect,
    onDisconnect,
    onError,
  } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [events, setEvents] = useState<LiveEvent[]>([]);
  const socketRef = useRef<Socket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;
  const baseDelay = 1000; // Start with 1 second

  const connect = useCallback(() => {
    if (socketRef.current?.connected) {
      return;
    }

    // Clean up existing socket if any
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
    }

    const socket = io(WS_URL, {
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionAttempts: maxReconnectAttempts,
      reconnectionDelay: baseDelay,
      reconnectionDelayMax: 10000, // Max 10 seconds
      timeout: 20000,
      // Add authentication headers if API key is provided
      extraHeaders: apiKey ? {
        'X-API-Key': apiKey,
      } : undefined,
      auth: apiKey ? {
        apiKey,
        organizationId,
      } : organizationId ? {
        organizationId,
      } : undefined,
    });

    socket.on('connect', () => {
      setIsConnected(true);
      reconnectAttempts.current = 0;
      logger.debug('WebSocket connected', { socketId: socket.id });
      onConnect?.();

      // Subscribe to organization if provided
      if (organizationId) {
        socket.emit('subscribe', organizationId);
        logger.debug('Subscribed to organization', { organizationId });
      }
    });

    socket.on('disconnect', (reason) => {
      setIsConnected(false);
      logger.debug('WebSocket disconnected', { reason });
      onDisconnect?.();

      // If disconnected due to server error, try to reconnect
      if (reason === 'io server disconnect') {
        // Server disconnected the socket, reconnect manually
        socket.connect();
      }
    });

    socket.on('connect_error', (error) => {
      reconnectAttempts.current++;
      const errorMessage = error.message || 'Failed to connect to WebSocket server';
      logger.error('WebSocket connection error', { 
        attempt: reconnectAttempts.current,
        maxAttempts: maxReconnectAttempts,
        error: errorMessage 
      });

      if (reconnectAttempts.current >= maxReconnectAttempts) {
        const finalError = new Error(
          `Failed to connect after ${maxReconnectAttempts} attempts: ${errorMessage}`
        );
        onError?.(finalError);
        // Stop trying to reconnect
        socket.disconnect();
      }
    });

    // Handle authentication errors
    socket.on('error', (error: any) => {
      logger.error('WebSocket error', { error });
      if (error.message?.includes('authentication') || error.message?.includes('API key')) {
        onError?.(new Error('WebSocket authentication failed. Please check your API key.'));
        socket.disconnect();
      }
    });

    // Handle subscription confirmations
    socket.on('subscribed', (data: { organizationId: string }) => {
      logger.debug('Successfully subscribed', { organizationId: data.organizationId });
    });

    socket.on('unsubscribed', (data: { organizationId: string }) => {
      logger.debug('Successfully unsubscribed', { organizationId: data.organizationId });
    });

    // Listen for threat detection events
    socket.on('threat_detected', (data: any) => {
      try {
        const event: LiveEvent = {
          id: data.id || `threat-${Date.now()}-${Math.random()}`,
          type: 'threat_detected',
          message: data.message || data.title || 'Threat detected',
          timestamp: data.timestamp ? new Date(data.timestamp) : new Date(),
          data,
        };
        setEvents((prev) => [event, ...prev.slice(0, 99)]); // Keep last 100 events
        logger.debug('Received threat_detected event', { eventId: event.id });
      } catch (error) {
        logger.error('Error processing threat_detected event', { error, data });
      }
    });

    socket.on('url_analyzed', (data: any) => {
      try {
        const event: LiveEvent = {
          id: data.id || `url-${Date.now()}-${Math.random()}`,
          type: 'detection',
          message: data.message || `URL analyzed: ${data.url || 'unknown'}`,
          timestamp: data.timestamp ? new Date(data.timestamp) : new Date(),
          data,
        };
        setEvents((prev) => [event, ...prev.slice(0, 99)]);
        logger.debug('Received url_analyzed event', { eventId: event.id });
      } catch (error) {
        logger.error('Error processing url_analyzed event', { error, data });
      }
    });

    socket.on('email_analyzed', (data: any) => {
      try {
        const event: LiveEvent = {
          id: data.id || `email-${Date.now()}-${Math.random()}`,
          type: 'detection',
          message: data.message || 'Email analyzed',
          timestamp: data.timestamp ? new Date(data.timestamp) : new Date(),
          data,
        };
        setEvents((prev) => [event, ...prev.slice(0, 99)]);
        logger.debug('Received email_analyzed event', { eventId: event.id });
      } catch (error) {
        logger.error('Error processing email_analyzed event', { error, data });
      }
    });

    // Generic event handler for other events
    socket.onAny((eventName, data) => {
      const ignoredEvents = [
        'connect', 
        'disconnect', 
        'connect_error', 
        'error',
        'threat_detected', 
        'url_analyzed', 
        'email_analyzed',
        'subscribed',
        'unsubscribed',
        'pong'
      ];
      
      if (!ignoredEvents.includes(eventName)) {
        try {
          const event: LiveEvent = {
            id: data.id || `${eventName}-${Date.now()}-${Math.random()}`,
            type: eventName as LiveEvent['type'],
            message: data.message || `Event: ${eventName}`,
            timestamp: data.timestamp ? new Date(data.timestamp) : new Date(),
            data,
          };
          setEvents((prev) => [event, ...prev.slice(0, 99)]);
          logger.debug('Received generic event', { eventName, eventId: event.id });
        } catch (error) {
          logger.error('Error processing generic event', { error, eventName, data });
        }
      }
    });

    socketRef.current = socket;
  }, [organizationId, apiKey, onConnect, onDisconnect, onError]);

  const disconnect = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
      setIsConnected(false);
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
  }, []);

  const subscribe = useCallback((orgId: string) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit('subscribe', orgId);
    }
  }, []);

  const unsubscribe = useCallback((orgId: string) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit('unsubscribe', orgId);
    }
  }, []);

  const clearEvents = useCallback(() => {
    setEvents([]);
  }, []);

  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  return {
    socket: socketRef.current,
    isConnected,
    events,
    connect,
    disconnect,
    subscribe,
    unsubscribe,
    clearEvents,
  };
}
