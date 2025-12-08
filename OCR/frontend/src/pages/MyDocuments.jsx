import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { FileText, Zap, Trash2, Loader2, Search } from 'lucide-react';
import { ocrAPI } from '../api/client';
import Sidebar from '../components/Sidebar';

export default function MyDocuments() {
    const [documents, setDocuments] = useState([]);
    const [loading, setLoading] = useState(true);
    const [searchQuery, setSearchQuery] = useState('');

    const navigate = useNavigate();

    const fetchDocuments = async () => {
        try {
            const response = await ocrAPI.getDocuments(0, 100);
            setDocuments(response.data.documents);
        } catch (error) {
            console.error('Failed to fetch documents:', error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchDocuments();
    }, []);

    const handleProcess = async (docId) => {
        try {
            await ocrAPI.process(docId);
            await fetchDocuments();
        } catch (error) {
            alert(error.response?.data?.detail || 'OCR failed');
        }
    };

    const handleDelete = async (docId) => {
        if (!confirm('Delete this document?')) return;
        try {
            await ocrAPI.deleteDocument(docId);
            await fetchDocuments();
        } catch (error) {
            alert(error.response?.data?.detail || 'Delete failed');
        }
    };

    const getStatusBadge = (status) => {
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

    const filteredDocs = documents.filter(doc =>
        doc.filename.toLowerCase().includes(searchQuery.toLowerCase())
    );

    if (loading) {
        return (
            <div className="min-h-screen flex items-center justify-center" style={{ background: '#0d0d0d' }}>
                <Loader2 className="w-6 h-6 text-white animate-spin" />
            </div>
        );
    }

    return (
        <div className="flex" style={{ background: '#0d0d0d', minHeight: '100vh' }}>
            <Sidebar recentDocs={documents} />

            <main className="flex-1 ml-[260px]">
                <div className="p-8">
                    {/* Header */}
                    <div className="mb-8">
                        <h1 className="text-xl font-semibold text-white mb-1">My Documents</h1>
                        <p className="text-gray-500 text-sm">
                            {documents.length} document{documents.length !== 1 ? 's' : ''} in your library
                        </p>
                    </div>

                    {/* Search */}
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

                    {/* Documents grid */}
                    {filteredDocs.length === 0 ? (
                        <div className="text-center py-12 text-gray-500 text-sm">
                            {documents.length === 0
                                ? 'No documents yet.'
                                : 'No documents match your search.'
                            }
                        </div>
                    ) : (
                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                            {filteredDocs.map((doc) => (
                                <div
                                    key={doc.id}
                                    className="card-white cursor-pointer hover:shadow-md transition"
                                    onClick={() => navigate(`/documents/${doc.id}`)}
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
                                                    onClick={(e) => { e.stopPropagation(); navigate(`/documents/${doc.id}`); }}
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
            </main>
        </div>
    );
}
