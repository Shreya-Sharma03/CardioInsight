/**
 * API client for CardioInsight FastAPI Backend.
 * Requests are forwarded via Vite proxy during development (/api -> http://127.0.0.1:8000).
 */
import {
  HealthResponse,
  RecordsListResponse,
  PredictRequest,
  PredictResponse,
  RawPredictResponse,
  BenchmarksResponse,
  ModelInfoResponse,
} from '../types';

const API_BASE = '/api';

class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  try {
    const res = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
      ...options,
    });

    if (!res.ok) {
      let errorMsg = `HTTP Error ${res.status}: ${res.statusText}`;
      try {
        const errorJson = await res.json();
        if (errorJson.detail) {
          errorMsg = typeof errorJson.detail === 'string'
            ? errorJson.detail
            : JSON.stringify(errorJson.detail);
        }
      } catch {
        // Fallback to status text
      }
      throw new ApiError(errorMsg, res.status);
    }

    return (await res.json()) as T;
  } catch (err: unknown) {
    if (err instanceof ApiError) {
      throw err;
    }
    const message = err instanceof Error ? err.message : 'Network request failed';
    throw new ApiError(`Failed to connect to CardioInsight API: ${message}`, 0);
  }
}

export async function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health');
}

export async function fetchRecords(): Promise<RecordsListResponse> {
  return request<RecordsListResponse>('/records');
}

export async function postPredict(
  recordIndex: number,
  threshold: number = 0.41
): Promise<PredictResponse> {
  const payload: PredictRequest = {
    record_index: recordIndex,
    threshold,
  };
  return request<PredictResponse>('/predict', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function postPredictRawRecord(
  recordIndex: number,
  threshold: number = 0.41
): Promise<RawPredictResponse> {
  const payload: PredictRequest = {
    record_index: recordIndex,
    threshold,
  };
  return request<RawPredictResponse>('/predict/raw-record', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function fetchBenchmarks(): Promise<BenchmarksResponse> {
  return request<BenchmarksResponse>('/benchmarks');
}

export async function fetchModelInfo(): Promise<ModelInfoResponse> {
  return request<ModelInfoResponse>('/model-info');
}
