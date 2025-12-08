import axios from 'axios';

// API base URL
const API_URL = 'http://localhost:8000';

// Create axios instance
const api = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Add token to requests
api.interceptors.request.use(
    (config) => {
        const token = localStorage.getItem('token');
        if (token) {
            config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
    },
    (error) => Promise.reject(error)
);

// Handle auth errors
api.interceptors.response.use(
    (response) => response,
    (error) => {
        if (error.response?.status === 401) {
            localStorage.removeItem('token');
            window.location.href = '/login';
        }
        return Promise.reject(error);
    }
);

// Auth API
export const authAPI = {
    login: async (username, password) => {
        const formData = new URLSearchParams();
        formData.append('username', username);
        formData.append('password', password);

        const response = await api.post('/auth/login', formData, {
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        });
        return response.data;
    },
};

// User API
export const userAPI = {
    getMe: () => api.get('/users/me'),
    getQuota: () => api.get('/users/quota'),
    getUsage: () => api.get('/users/usage'),
};

// OCR API
export const ocrAPI = {
    upload: async (file) => {
        const formData = new FormData();
        formData.append('file', file);

        return api.post('/ocr/upload', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
        });
    },

    process: (documentId, options = {}) =>
        api.post(`/ocr/process/${documentId}`, {
            language: options.language || 'eng',
            preprocess: options.preprocess || false,
            use_hybrid: options.useHybrid || false,
            min_confidence: options.minConfidence || 0.75,
        }),

    extract: (documentId, schemaPrompt, minConfidence = 0.75) =>
        api.post(`/ocr/extract/${documentId}`, {
            schema_prompt: schemaPrompt,
            min_confidence: minConfidence,
        }),

    getDocuments: (skip = 0, limit = 20) =>
        api.get(`/ocr/documents?skip=${skip}&limit=${limit}`),

    getDocument: (documentId) =>
        api.get(`/ocr/documents/${documentId}`),

    getText: (documentId) =>
        api.get(`/ocr/documents/${documentId}/text`),

    getToon: (documentId) =>
        api.get(`/ocr/documents/${documentId}/toon`),

    getUsage: (documentId) =>
        api.get(`/ocr/documents/${documentId}/usage`),

    deleteDocument: (documentId) =>
        api.delete(`/ocr/documents/${documentId}`),
};

export default api;
