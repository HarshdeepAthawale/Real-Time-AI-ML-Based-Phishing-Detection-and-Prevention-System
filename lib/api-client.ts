import axios, { AxiosInstance, AxiosError, AxiosRequestConfig } from 'axios';
import { ApiError } from './types/api';

// Get API URL from env (build-time) or localStorage (runtime override)
const DEFAULT_API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3000';

export function getApiBaseUrl(): string {
  if (typeof window !== 'undefined') {
    const stored = localStorage.getItem('api_url');
    if (stored?.trim()) return stored.trim();
  }
  return DEFAULT_API_URL;
}

// Create axios instance - baseURL set per-request via interceptor for runtime override
const apiClient: AxiosInstance = axios.create({
  baseURL: DEFAULT_API_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor - Add auth token and runtime API URL override
apiClient.interceptors.request.use(
  (config) => {
    config.baseURL = getApiBaseUrl();
    const apiKey = typeof window !== 'undefined' ? localStorage.getItem('api_key') : null;
    if (apiKey) {
      config.headers['X-API-Key'] = apiKey;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor - Handle errors globally
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error: AxiosError) => {
    const apiError: ApiError = {
      error: 'Request failed',
      message: error.message,
      statusCode: error.response?.status,
      details: error.response?.data,
    };

    // Handle specific error cases
    if (error.response) {
      // Server responded with error status
      const data = error.response.data as any;
      apiError.error = data.error || 'Server error';
      apiError.message = data.message || error.message;
      apiError.details = data.details || data;

      // Handle 401 Unauthorized - API key missing or invalid
      if (error.response.status === 401) {
        apiError.error = 'Authentication required';
        apiError.message = 'API key is missing or invalid. Please configure your API key.';
        // Optionally clear invalid API key from localStorage
        if (typeof window !== 'undefined') {
          const storedKey = localStorage.getItem('api_key');
          if (storedKey) {
            console.warn('API key authentication failed. Consider updating your API key.');
          }
        }
      }

      // Handle 403 Forbidden
      if (error.response.status === 403) {
        apiError.error = 'Access forbidden';
        apiError.message = 'You do not have permission to access this resource.';
      }

      // Handle 404 Not Found
      if (error.response.status === 404) {
        apiError.error = 'Resource not found';
        apiError.message = 'The requested resource was not found.';
      }

      // Handle 500+ Server errors
      if (error.response.status >= 500) {
        apiError.error = 'Server error';
        apiError.message = 'The server encountered an error. Please try again later.';
      }
    } else if (error.request) {
      // Request made but no response received
      apiError.error = 'Network error';
      apiError.message = 'Unable to reach server. Please check your connection and ensure the API is running.';
    }

    return Promise.reject(apiError);
  }
);

// Generic API request function
export async function apiRequest<T>(
  config: AxiosRequestConfig
): Promise<T> {
  try {
    const response = await apiClient.request<T>(config);
    return response.data;
  } catch (error) {
    throw error;
  }
}

// GET request helper
export async function apiGet<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
  return apiRequest<T>({ ...config, method: 'GET', url });
}

// POST request helper
export async function apiPost<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
  return apiRequest<T>({ ...config, method: 'POST', url, data });
}

// PUT request helper
export async function apiPut<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
  return apiRequest<T>({ ...config, method: 'PUT', url, data });
}

// DELETE request helper
export async function apiDelete<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
  return apiRequest<T>({ ...config, method: 'DELETE', url });
}

export default apiClient;
