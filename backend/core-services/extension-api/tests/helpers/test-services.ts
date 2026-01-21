import axios from 'axios';

/**
 * Check if detection-api service is available
 */
export const isDetectionApiAvailable = async (url?: string): Promise<boolean> => {
  const detectionApiUrl = url || process.env.DETECTION_API_URL || 'http://localhost:3001';
  
  try {
    const response = await axios.get(`${detectionApiUrl}/health`, {
      timeout: 2000,
    });
    return response.status === 200;
  } catch {
    return false;
  }
};

/**
 * Check if threat-intel service is available
 */
export const isThreatIntelAvailable = async (url?: string): Promise<boolean> => {
  const threatIntelUrl = url || process.env.THREAT_INTEL_URL || 'http://localhost:3002';
  
  try {
    const response = await axios.get(`${threatIntelUrl}/health`, {
      timeout: 2000,
    });
    return response.status === 200;
  } catch {
    return false;
  }
};

/**
 * Skip test if service is not available
 */
export const skipIfServiceUnavailable = (
  serviceName: string,
  checkFn: () => Promise<boolean>
): void => {
  beforeAll(async () => {
    const isAvailable = await checkFn();
    if (!isAvailable) {
      console.warn(`Skipping tests - ${serviceName} is not available`);
      pending();
    }
  });
};

/**
 * Wait for service to be available (with timeout)
 */
export const waitForService = async (
  checkFn: () => Promise<boolean>,
  timeout: number = 30000,
  interval: number = 1000
): Promise<boolean> => {
  const startTime = Date.now();
  
  while (Date.now() - startTime < timeout) {
    if (await checkFn()) {
      return true;
    }
    await new Promise(resolve => setTimeout(resolve, interval));
  }
  
  return false;
};
