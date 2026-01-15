import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE,
});

export const uploadVideo = async (file) => {
  const formData = new FormData();
  formData.append('video', file);
  return api.post('/video/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

export const downloadVideo = async (filename) => {
  return api.get(`/video/download/${filename}`, { responseType: 'blob' });
};

export const getJobStatus = async (jobId) => {
  return api.get(`/video/status/${jobId}`);
};

export const checkHealth = async () => {
  // baseURL already ends with /api, so this should be /health
  return api.get('/health');
};

export default api;
