'use client';

import { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { Upload, FileText, Zap, Trash2, Loader2, Plus, Search } from 'lucide-react';
import { ocrAPI, userAPI } from '@/lib/api';
import { useAuth } from '@/context/AuthContext';
import Sidebar from '@/components/Sidebar';

interface Document {
  id: number;
  filename: string;
  status: string;
  ocr_confidence?: number;
}

interface Quota {
  tokens_used_today: number;
  max_tokens_daily: number;
}

export default function DashboardPage() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [quota, setQuota] = useState<Quota | null>(null);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const router = useRouter();
  const { user, isAuthenticated, loading: authLoading } = useAuth();

  const fetchData = useCallback(async () => {
    try {
      const [docsRes, quotaRes] = await Promise.all([
        ocrAPI.getDocuments(),
        userAPI.getQuota(),
      ]);
      setDocuments(docsRes.data.documents);
      setQuota(quotaRes.data);
    } catch (error) {
      console.error('Failed to fetch data:', error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      router.push('/login');
      return;
    }
    if (isAuthenticated) {
      fetchData();
    }
  }, [fetchData, isAuthenticated, authLoading, router]);

  const handleUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;

    setUploading(true);
    try {
      for (const file of Array.from(files)) {
        await ocrAPI.upload(file);
      }
      await fetchData();
    } catch (error: any) {
      console.error('Upload failed:', error);
      alert(error.response?.data?.detail || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    handleUpload(e.dataTransfer.files);
  };

  const handleProcess = async (docId: number) => {
    try {
      await ocrAPI.process(docId);
      await fetchData();
    } catch (error: any) {
      alert(error.response?.data?.detail || 'OCR failed');
    }
  };

  const handleDelete = async (docId: number) => {
    if (!confirm('Delete this document?')) return;
    try {
      await ocrAPI.deleteDocument(docId);
      await fetchData();
    } catch (error: any) {
      alert(error.response?.data?.detail || 'Delete failed');
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed':
        return <span className="text-xs px-2 py-0.5 rounded bg-green-100 text-green-700">Completed</span>;
      case 'processing':
        return <span className="text-xs px-2 py-0.5 rounded bg-blue-100 text-blue-700">Processing</span>;
      case 'failed':
        return <span className="text-xs px-2 py-0.5 rounded bg-red-100 text-red-700">Failed</span>;
      default:
        return <span className="text-xs px-2 py-0.5 rounded bg-yellow-100 text-yellow-700">Pending</span>;
    }
  };

  const formatNumber = (num: number) => {
    if (num >= 1000000000) return `${(num / 1000000000).toFixed(0)}B`;
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(0)}K`;
    return num.toLocaleString();
  };

  if (loading || authLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ background: '#0d0d0d' }}>
        <Loader2 className="w-6 h-6 text-white animate-spin" />
      </div>
    );
  }

  const filteredDocs = documents.filter(doc =>
    doc.filename.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="flex" style={{ background: '#0d0d0d', minHeight: '100vh' }}>
      <Sidebar recentDocs={documents} />

      <main className="flex-1 ml-[260px]">
        <div className="p-8">
          {/* Header */}
          <div className="flex items-center justify-between mb-8">
            <div>
              <h1 className="text-xl font-semibold text-white mb-1">
                Hi, {user?.username || 'User'}
              </h1>
              <p className="text-gray-500 text-sm">
                Upload and process your documents
              </p>
            </div>

            <button
              className="btn-primary"
              onClick={() => document.getElementById('fileInput')?.click()}
            >
              <Plus className="w-4 h-4" />
              New Upload
            </button>
          </div>

          {/* Search bar */}
          <div className="mb-8 max-w-xl">
            <div className="relative">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-600" />
              <input
                type="text"
                placeholder="Search documents..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="input-dark pl-11"
              />
            </div>
          </div>

          {/* Upload zone */}
          <div
            className={`border-2 border-dashed rounded-xl p-10 mb-8 text-center cursor-pointer transition ${dragOver
              ? 'border-blue-500 bg-blue-500/5'
              : 'border-[#2a2a2a] hover:border-[#3a3a3a]'
              }`}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => document.getElementById('fileInput')?.click()}
          >
            <input
              id="fileInput"
              type="file"
              multiple
              accept=".pdf,.png,.jpg,.jpeg,.bmp,.tiff"
              className="hidden"
              onChange={(e) => handleUpload(e.target.files)}
            />

            {uploading ? (
              <Loader2 className="w-10 h-10 text-gray-500 mx-auto mb-3 animate-spin" />
            ) : (
              <Upload className="w-10 h-10 text-gray-500 mx-auto mb-3" />
            )}

            <p className="text-white mb-1">
              {uploading ? 'Uploading...' : 'Drop files here or click to upload'}
            </p>
            <p className="text-gray-500 text-sm">
              Supports: PDF, PNG, JPG, TIFF
            </p>
          </div>

          {/* Documents section */}
          <div>
            <h2 className="text-base font-medium text-white mb-2">
              My Documents
            </h2>
            <p className="text-gray-500 text-sm mb-6">
              {documents.length} document{documents.length !== 1 ? 's' : ''} in your library
            </p>

            {filteredDocs.length === 0 ? (
              <div className="text-center py-12 text-gray-500 text-sm">
                {documents.length === 0
                  ? 'No documents yet. Upload your first file!'
                  : 'No documents match your search.'
                }
              </div>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {filteredDocs.map((doc) => (
                  <div
                    key={doc.id}
                    className="card-white cursor-pointer hover:shadow-md transition"
                    onClick={() => router.push(`/documents/${doc.id}`)}
                  >
                    <div className="flex items-start gap-3 mb-3">
                      <div className="w-9 h-9 rounded-lg bg-blue-50 flex items-center justify-center flex-shrink-0">
                        <FileText className="w-4 h-4 text-blue-600" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <h3 className="font-medium text-gray-900 text-sm truncate">
                          {doc.filename}
                        </h3>
                        <p className="text-xs text-gray-500 mt-0.5">
                          {doc.ocr_confidence
                            ? `${(doc.ocr_confidence * 100).toFixed(0)}% confidence`
                            : 'Not processed'
                          }
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center justify-between">
                      {getStatusBadge(doc.status)}

                      <div className="flex items-center gap-1">
                        {doc.status === 'pending' && (
                          <button
                            onClick={(e) => { e.stopPropagation(); handleProcess(doc.id); }}
                            className="text-xs px-2 py-1 text-blue-600 hover:bg-blue-50 rounded transition"
                          >
                            OCR
                          </button>
                        )}

                        {doc.status === 'completed' && (
                          <button
                            onClick={(e) => { e.stopPropagation(); router.push(`/documents/${doc.id}`); }}
                            className="text-xs px-2 py-1 text-purple-600 hover:bg-purple-50 rounded transition"
                          >
                            Extract
                          </button>
                        )}

                        <button
                          onClick={(e) => { e.stopPropagation(); handleDelete(doc.id); }}
                          className="p-1 text-gray-400 hover:text-red-500 transition"
                        >
                          <Trash2 className="w-3 h-3" />
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Quota info */}
        {quota && (
          <div className="fixed bottom-4 right-4 text-xs text-gray-600">
            {formatNumber(quota.tokens_used_today)} / {formatNumber(quota.max_tokens_daily)} tokens today
          </div>
        )}
      </main>
    </div>
  );
}
