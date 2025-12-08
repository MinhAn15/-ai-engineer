import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
    ArrowLeft, FileText, Zap, Copy, Check,
    Loader2, Code, Database, Send, Upload, CheckCircle, Clock, Coins
} from 'lucide-react';
import { ocrAPI } from '../api/client';
import Sidebar from '../components/Sidebar';

export default function DocumentDetail() {
    const { id } = useParams();
    const navigate = useNavigate();

    const [document, setDocument] = useState(null);
    const [ocrText, setOcrText] = useState('');
    const [toonOutput, setToonOutput] = useState('');
    const [usage, setUsage] = useState(null);
    const [schemaPrompt, setSchemaPrompt] = useState('Extract: company name, date, invoice number, total amount');
    const [loading, setLoading] = useState(true);
    const [processing, setProcessing] = useState(false);
    const [extracting, setExtracting] = useState(false);
    const [copied, setCopied] = useState(false);
    const [activeTab, setActiveTab] = useState('ocr');
    const [useHybrid, setUseHybrid] = useState(false);  // Hybrid OCR toggle

    useEffect(() => {
        fetchDocument();
    }, [id]);

    const fetchDocument = async () => {
        try {
            const docRes = await ocrAPI.getDocument(id);
            setDocument(docRes.data);

            // Fetch usage/pipeline data
            try {
                const usageRes = await ocrAPI.getUsage(id);
                setUsage(usageRes.data);
            } catch (e) { }

            if (docRes.data.status === 'completed') {
                try {
                    const textRes = await ocrAPI.getText(id);
                    setOcrText(textRes.data.text);
                } catch (e) { }

                try {
                    const toonRes = await ocrAPI.getToon(id);
                    setToonOutput(toonRes.data.toon);
                } catch (e) { }
            }
        } catch (error) {
            console.error('Failed to fetch document:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleProcess = async () => {
        setProcessing(true);
        try {
            await ocrAPI.process(id, { useHybrid: useHybrid });
            await fetchDocument();
        } catch (error) {
            alert(error.response?.data?.detail || 'OCR failed');
        } finally {
            setProcessing(false);
        }
    };

    const handleExtract = async () => {
        if (!schemaPrompt.trim()) {
            alert('Please enter a schema prompt');
            return;
        }

        setExtracting(true);
        try {
            const response = await ocrAPI.extract(id, schemaPrompt);
            setToonOutput(response.data.toon_output);
            setActiveTab('toon');
            await fetchDocument();
        } catch (error) {
            alert(error.response?.data?.detail || 'Extraction failed');
        } finally {
            setExtracting(false);
        }
    };

    const copyToClipboard = (text) => {
        navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    const PipelineStep = ({ icon: Icon, title, status, details }) => {
        const isCompleted = status === 'completed';
        const isPending = status === 'pending';

        return (
            <div className={`flex items-center gap-3 p-3 rounded-lg ${isCompleted ? 'bg-green-50' : isPending ? 'bg-gray-50' : 'bg-red-50'}`}>
                <div className={`w-8 h-8 rounded-full flex items-center justify-center ${isCompleted ? 'bg-green-100' : isPending ? 'bg-gray-200' : 'bg-red-100'
                    }`}>
                    {isCompleted ? (
                        <CheckCircle className="w-4 h-4 text-green-600" />
                    ) : isPending ? (
                        <Clock className="w-4 h-4 text-gray-400" />
                    ) : (
                        <Icon className="w-4 h-4 text-red-600" />
                    )}
                </div>
                <div className="flex-1">
                    <div className="text-sm font-medium text-gray-900">{title}</div>
                    {details && <div className="text-xs text-gray-500">{details}</div>}
                </div>
            </div>
        );
    };

    if (loading) {
        return (
            <div className="min-h-screen flex items-center justify-center" style={{ background: '#0d0d0d' }}>
                <Loader2 className="w-6 h-6 text-white animate-spin" />
            </div>
        );
    }

    if (!document) {
        return (
            <div className="min-h-screen flex items-center justify-center text-white" style={{ background: '#0d0d0d' }}>
                Document not found
            </div>
        );
    }

    return (
        <div className="flex" style={{ background: '#0d0d0d', minHeight: '100vh' }}>
            <Sidebar />

            <main className="flex-1 ml-[260px] p-8">
                {/* Header */}
                <div className="flex items-center gap-4 mb-6">
                    <button
                        onClick={() => navigate('/')}
                        className="p-2 text-gray-400 hover:text-white transition rounded-lg hover:bg-white/5"
                    >
                        <ArrowLeft className="w-5 h-5" />
                    </button>

                    <div className="w-10 h-10 rounded-lg bg-blue-100 flex items-center justify-center">
                        <FileText className="w-5 h-5 text-blue-600" />
                    </div>

                    <div className="flex-1">
                        <h1 className="text-lg font-semibold text-white">{document.filename}</h1>
                        <p className="text-gray-500 text-sm">
                            {document.ocr_confidence
                                ? `${(document.ocr_confidence * 100).toFixed(1)}% confidence`
                                : document.status
                            }
                        </p>
                    </div>

                    {/* Token usage badge */}
                    {usage && usage.total_tokens > 0 && (
                        <div className="flex items-center gap-2 px-3 py-1.5 bg-yellow-500/10 rounded-lg">
                            <Coins className="w-4 h-4 text-yellow-500" />
                            <span className="text-yellow-500 text-sm font-medium">
                                {usage.total_tokens.toLocaleString()} tokens
                            </span>
                        </div>
                    )}
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Left panel - Pipeline & Actions */}
                    <div className="space-y-6">
                        {/* Pipeline Visualization */}
                        {usage?.pipeline && (
                            <div className="card-white">
                                <h2 className="font-semibold text-gray-900 mb-4">Processing Pipeline</h2>
                                <div className="space-y-2">
                                    <PipelineStep
                                        icon={Upload}
                                        title="Upload"
                                        status={usage.pipeline.upload.status}
                                        details={usage.pipeline.upload.filename}
                                    />
                                    <div className="flex justify-center">
                                        <div className="w-px h-4 bg-gray-200"></div>
                                    </div>
                                    <PipelineStep
                                        icon={Zap}
                                        title="OCR Processing"
                                        status={usage.pipeline.ocr.status}
                                        details={usage.pipeline.ocr.confidence
                                            ? `${(usage.pipeline.ocr.confidence * 100).toFixed(0)}% • ${usage.pipeline.ocr.text_blocks} blocks • ${usage.pipeline.ocr.processing_time_ms?.toFixed(0)}ms`
                                            : null
                                        }
                                    />
                                    <div className="flex justify-center">
                                        <div className="w-px h-4 bg-gray-200"></div>
                                    </div>
                                    <PipelineStep
                                        icon={Send}
                                        title="LLM Extraction"
                                        status={usage.pipeline.extraction.status}
                                        details={usage.pipeline.extraction.entities
                                            ? `${usage.pipeline.extraction.entities} entities • ${usage.pipeline.extraction.relations} relations`
                                            : null
                                        }
                                    />
                                </div>

                                {/* Token cost summary */}
                                {usage.total_tokens > 0 && (
                                    <div className="mt-4 pt-4 border-t border-gray-100">
                                        <div className="flex justify-between text-sm">
                                            <span className="text-gray-500">Total tokens used</span>
                                            <span className="font-medium text-gray-900">{usage.total_tokens.toLocaleString()}</span>
                                        </div>
                                        <div className="flex justify-between text-sm mt-1">
                                            <span className="text-gray-500">Est. cost</span>
                                            <span className="font-medium text-gray-900">${usage.total_cost_usd.toFixed(6)}</span>
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}

                        {/* OCR Action */}
                        {document.status === 'pending' && (
                            <div className="card-white">
                                <h2 className="font-medium text-gray-900 mb-3">Run OCR</h2>

                                {/* Hybrid OCR Toggle */}
                                <div className="flex items-center justify-between mb-4 p-3 bg-gray-50 rounded-lg">
                                    <div>
                                        <div className="text-sm font-medium text-gray-900">Hybrid OCR</div>
                                        <div className="text-xs text-gray-500">
                                            Use AI for low-confidence regions
                                        </div>
                                    </div>
                                    <button
                                        type="button"
                                        onClick={() => setUseHybrid(!useHybrid)}
                                        className={`relative w-11 h-6 rounded-full transition-colors ${useHybrid ? 'bg-blue-500' : 'bg-gray-300'
                                            }`}
                                    >
                                        <span
                                            className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform ${useHybrid ? 'translate-x-5' : 'translate-x-0'
                                                }`}
                                        />
                                    </button>
                                </div>

                                {useHybrid && (
                                    <div className="mb-3 p-2 bg-yellow-50 border border-yellow-200 rounded-lg text-xs text-yellow-700">
                                        ⚠️ Hybrid mode uses AI tokens for low-confidence text
                                    </div>
                                )}

                                <button
                                    onClick={handleProcess}
                                    disabled={processing}
                                    className="w-full py-2 bg-blue-500 text-white rounded-lg text-sm font-medium flex items-center justify-center gap-2 hover:bg-blue-600 transition disabled:opacity-50"
                                >
                                    {processing ? (
                                        <>
                                            <Loader2 className="w-4 h-4 animate-spin" />
                                            Processing...
                                        </>
                                    ) : (
                                        <>
                                            <Zap className="w-4 h-4" />
                                            {useHybrid ? 'Start Hybrid OCR' : 'Start OCR'}
                                        </>
                                    )}
                                </button>
                            </div>
                        )}

                        {/* Extract Action */}
                        {document.status === 'completed' && (
                            <div className="card-white">
                                <h2 className="font-medium text-gray-900 mb-3">Schema Extraction</h2>

                                <textarea
                                    value={schemaPrompt}
                                    onChange={(e) => setSchemaPrompt(e.target.value)}
                                    className="w-full border border-gray-200 rounded-lg p-2 text-gray-900 text-sm focus:outline-none focus:border-purple-500 resize-none"
                                    rows={3}
                                    placeholder="e.g., Extract: company name, invoice number"
                                />

                                <button
                                    onClick={handleExtract}
                                    disabled={extracting}
                                    className="mt-3 w-full py-2 bg-purple-500 text-white rounded-lg text-sm font-medium flex items-center justify-center gap-2 hover:bg-purple-600 transition disabled:opacity-50"
                                >
                                    {extracting ? (
                                        <>
                                            <Loader2 className="w-4 h-4 animate-spin" />
                                            Extracting...
                                        </>
                                    ) : (
                                        <>
                                            <Send className="w-4 h-4" />
                                            Extract
                                        </>
                                    )}
                                </button>
                            </div>
                        )}

                        {/* Stats */}
                        {document.num_entities > 0 && (
                            <div className="card-white">
                                <h2 className="font-medium text-gray-900 mb-3">Extraction Results</h2>
                                <div className="grid grid-cols-2 gap-3">
                                    <div className="text-center p-3 bg-gray-50 rounded-lg">
                                        <div className="text-xl font-bold text-gray-900">{document.num_entities}</div>
                                        <div className="text-xs text-gray-500">Entities</div>
                                    </div>
                                    <div className="text-center p-3 bg-gray-50 rounded-lg">
                                        <div className="text-xl font-bold text-gray-900">{document.num_relations}</div>
                                        <div className="text-xs text-gray-500">Relations</div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Right panel - Output */}
                    <div className="lg:col-span-2 card-dark">
                        {/* Tabs */}
                        <div className="flex gap-1 mb-4 p-1 bg-[#141414] rounded-lg inline-flex">
                            <button
                                onClick={() => setActiveTab('ocr')}
                                className={`px-4 py-2 rounded-md text-sm font-medium transition ${activeTab === 'ocr'
                                    ? 'bg-[#1e1e1e] text-white'
                                    : 'text-gray-400 hover:text-white'
                                    }`}
                            >
                                <Code className="w-4 h-4 inline mr-2" />
                                OCR Text
                            </button>
                            <button
                                onClick={() => setActiveTab('toon')}
                                className={`px-4 py-2 rounded-md text-sm font-medium transition ${activeTab === 'toon'
                                    ? 'bg-[#1e1e1e] text-white'
                                    : 'text-gray-400 hover:text-white'
                                    }`}
                            >
                                <Database className="w-4 h-4 inline mr-2" />
                                TOON Output
                            </button>
                        </div>

                        {/* Content */}
                        <div className="relative">
                            <button
                                onClick={() => copyToClipboard(activeTab === 'ocr' ? ocrText : toonOutput)}
                                className="absolute top-2 right-2 p-2 bg-white/10 rounded-lg text-gray-400 hover:text-white transition"
                            >
                                {copied ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4" />}
                            </button>

                            <pre className="bg-[#0d0d0d] rounded-lg p-4 h-[500px] overflow-auto text-sm font-mono">
                                <code className={activeTab === 'ocr' ? 'text-gray-300' : 'text-purple-300'}>
                                    {activeTab === 'ocr'
                                        ? (ocrText || 'Run OCR first to see extracted text')
                                        : (toonOutput || 'Run extraction to see TOON output')
                                    }
                                </code>
                            </pre>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
}
