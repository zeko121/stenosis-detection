/**
 * API Client
 * Handles all HTTP requests to the backend API
 */
import axios, { AxiosInstance } from 'axios';
import type {
  CaseListItem,
  CaseDetail,
  UploadResponse,
  ProcessingStatus,
  CaseResults,
  VolumeMetadata,
} from '@/types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const API_V1 = `${API_BASE_URL}/api/v1`;

class APIClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_V1,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  // Cases API
  async uploadCase(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await this.client.post<UploadResponse>('/cases/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  }

  async listCases(params?: {
    skip?: number;
    limit?: number;
    status?: string;
  }): Promise<CaseListItem[]> {
    const response = await this.client.get<CaseListItem[]>('/cases', { params });
    return response.data;
  }

  async getCase(caseId: string): Promise<CaseDetail> {
    const response = await this.client.get<CaseDetail>(`/cases/${caseId}`);
    return response.data;
  }

  async processCase(caseId: string, config?: any): Promise<ProcessingStatus> {
    const response = await this.client.post<ProcessingStatus>(
      `/cases/${caseId}/process`,
      config || {}
    );
    return response.data;
  }

  async getCaseStatus(caseId: string): Promise<ProcessingStatus> {
    const response = await this.client.get<ProcessingStatus>(`/cases/${caseId}/status`);
    return response.data;
  }

  async deleteCase(caseId: string): Promise<void> {
    await this.client.delete(`/cases/${caseId}`);
  }

  // Results API
  async getCaseResults(caseId: string): Promise<CaseResults> {
    const response = await this.client.get<CaseResults>(`/results/${caseId}`);
    return response.data;
  }

  async generateReport(caseId: string, options?: {
    format?: 'pdf' | 'json';
    template?: 'brief' | 'comprehensive';
  }): Promise<{ download_url: string }> {
    const response = await this.client.post(`/results/${caseId}/report`, options || {});
    return response.data;
  }

  // Visualization API
  async getVolumeMetadata(caseId: string): Promise<VolumeMetadata> {
    const response = await this.client.get<VolumeMetadata>(
      `/visualization/${caseId}/metadata`
    );
    return response.data;
  }

  getSliceUrl(
    caseId: string,
    plane: 'axial' | 'coronal' | 'sagittal',
    index: number,
    options?: {
      window_min?: number;
      window_max?: number;
      show_overlay?: boolean;
    }
  ): string {
    const params = new URLSearchParams({
      plane,
      index: index.toString(),
      window_min: (options?.window_min || -100).toString(),
      window_max: (options?.window_max || 1000).toString(),
      show_overlay: (options?.show_overlay || false).toString(),
    });

    return `${API_V1}/visualization/${caseId}/slice?${params}`;
  }

  async downloadReport(caseId: string, filename: string): Promise<Blob> {
    const response = await this.client.get(`/results/${caseId}/download/${filename}`, {
      responseType: 'blob',
    });
    return response.data;
  }
}

export const apiClient = new APIClient();
