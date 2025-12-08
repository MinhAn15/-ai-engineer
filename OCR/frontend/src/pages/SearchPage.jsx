import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { FileText, Search as SearchIcon, Loader2 } from 'lucide-react';
import { ocrAPI } from '../api/client';
import Sidebar from '../components/Sidebar';

export default function SearchPage() {
    const [documents, setDocuments] = useState([]);
    const [loading, setLoading] = useState(true);
    const [searchQuery, setSearchQuery] = useState('');

    const navigate = useNavigate();

    useEffect(() => {
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
        fetchDocuments();
    }, []);

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
                        <h1 className="text-xl font-semibold text-white mb-1">Search</h1>
                        <p className="text-gray-500 text-sm">Find documents in your library</p>
                    </div>

                    {/* Search input */}
                    <div className="mb-8">
                        <div className="relative max-w-2xl">
                            <SearchIcon className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-600" />
                            <input
                                type="text"
                                placeholder="Search by filename..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                className="input-dark pl-12 text-lg py-4"
                                autoFocus
                            />
                        </div>
                    </div>

                    {/* Results */}
                    {searchQuery && (
                        <div>
                            <p className="text-gray-500 text-sm mb-4">
                                {filteredDocs.length} result{filteredDocs.length !== 1 ? 's' : ''} for "{searchQuery}"
                            </p>

                            {filteredDocs.length === 0 ? (
                                <div className="text-center py-12 text-gray-500">
                                    No documents found.
                                </div>
                            ) : (
                                <div className="space-y-2">
                                    {filteredDocs.map((doc) => (
                                        <div
                                            key={doc.id}
                                            className="flex items-center gap-4 p-4 rounded-lg bg-[#1a1a1a] hover:bg-[#222] cursor-pointer transition"
                                            onClick={() => navigate(`/documents/${doc.id}`)}
                                        >
                                            <div className="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center">
                                                <FileText className="w-5 h-5 text-blue-400" />
                                            </div>
                                            <div className="flex-1">
                                                <h3 className="text-white font-medium">{doc.filename}</h3>
                                                <p className="text-gray-500 text-sm">
                                                    {doc.status} • {doc.ocr_confidence
                                                        ? `${(doc.ocr_confidence * 100).toFixed(0)}% confidence`
                                                        : 'Not processed'
                                                    }
                                                </p>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}

                    {/* Empty state */}
                    {!searchQuery && (
                        <div className="text-center py-20 text-gray-500">
                            <SearchIcon className="w-12 h-12 mx-auto mb-4 opacity-30" />
                            <p>Start typing to search documents</p>
                        </div>
                    )}
                </div>
            </main>
        </div>
    );
}
