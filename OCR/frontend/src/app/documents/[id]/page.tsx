'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useRouter } from 'next/navigation';
import {
    ArrowLeft, FileText, Zap, Copy, Check,
    Loader2, Code, Database, Send, Upload, CheckCircle, Clock, Coins, Image, List
} from 'lucide-react';
import { ocrAPI } from '@/lib/api';
import Sidebar from '@/components/Sidebar';

interface Segment {
    index: number;
    text: string;
    confidence: number;
    bbox: [number, number, number, number];
    type?: string;
    type_confidence?: number;
    color?: string;
}

interface ExtractedField {
    name: string;
    value: string;
    refs: number[];
}

interface DocumentType {
    id: number;
    filename: string;
    status: string;
    ocr_confidence?: number;
}

interface Usage {
    total_tokens: number;
    pipeline?: {
        upload: { status: string; filename: string };
        ocr: { status: string; confidence?: number; text_blocks?: number };
        extraction: { status: string; entities?: number; relations?: number };
    };
}

export default function DocumentDetailPage() {
    const params = useParams();
    const id = params.id as string;
    const router = useRouter();
    const containerRef = useRef<HTMLDivElement>(null);
    const imageRef = useRef<HTMLImageElement>(null);

    const [document, setDocument] = useState<DocumentType | null>(null);
    const [ocrText, setOcrText] = useState('');
    const [toonOutput, setToonOutput] = useState('');
    const [usage, setUsage] = useState<Usage | null>(null);
    const [segments, setSegments] = useState<Segment[]>([]);
    const [imageUrl, setImageUrl] = useState<string | null>(null);
    const [imageLoaded, setImageLoaded] = useState(false);
    const [hoveredSegmentIndex, setHoveredSegmentIndex] = useState<number | null>(null);
    const [schemaPrompt, setSchemaPrompt] = useState('Extract: company name, date, invoice number, total amount');
    const [loading, setLoading] = useState(true);
    const [processing, setProcessing] = useState(false);
    const [extracting, setExtracting] = useState(false);
    const [copied, setCopied] = useState(false);
    const [activeTab, setActiveTab] = useState('segments');
    const [useHybrid, setUseHybrid] = useState(false);
    const [extractedFields, setExtractedFields] = useState<ExtractedField[]>([]);
    const [minConfidence, setMinConfidence] = useState(0.75);
    const [classifying, setClassifying] = useState(false);
    const [classified, setClassified] = useState(false);
    const [tokenEstimate, setTokenEstimate] = useState<number | null>(null);
    const [colorLegend, setColorLegend] = useState<Record<string, string>>({});
    const [layoutSummary, setLayoutSummary] = useState<Record<string, number>>({});

    useEffect(() => {
        if (id) fetchDocument();
    }, [id]);

    const fetchDocument = async () => {
        try {
            const docRes = await ocrAPI.getDocument(Number(id));
            setDocument(docRes.data);

            try {
                const usageRes = await ocrAPI.getUsage(Number(id));
                setUsage(usageRes.data);
            } catch (e) { }

            if (docRes.data.status === 'completed') {
                try {
                    const textRes = await ocrAPI.getText(Number(id));
                    setOcrText(textRes.data.text);
                } catch (e) { }

                try {
                    const toonRes = await ocrAPI.getToon(Number(id));
                    setToonOutput(toonRes.data.toon);
                } catch (e) { }

                try {
                    const segRes = await ocrAPI.getSegments(Number(id));
                    setSegments(segRes.data.segments || []);
                    setColorLegend(segRes.data.color_legend || {});
                    setLayoutSummary(segRes.data.layout_summary || {});
                    setClassified(segRes.data.classified || false);
                } catch (e) {
                    console.error('Failed to fetch segments:', e);
                }

                try {
                    const imgUrl = await ocrAPI.getImage(Number(id));
                    setImageUrl(imgUrl);
                } catch (e) { }
            }
        } catch (error) {
            console.error('Failed to fetch document:', error);
        } finally {
            setLoading(false);
        }
    };

    const getHighlightStyle = useCallback((): React.CSSProperties => {
        if (hoveredSegmentIndex === null || !imageRef.current || !imageLoaded) {
            return { display: 'none' };
        }

        const segment = segments.find(s => s.index === hoveredSegmentIndex);
        if (!segment || !segment.bbox) {
            return { display: 'none' };
        }

        const [x, y, w, h] = segment.bbox;
        const img = imageRef.current;

        const scaleX = img.clientWidth / img.naturalWidth;
        const scaleY = img.clientHeight / img.naturalHeight;

        const segmentColor = segment.color || '#22c55e';
        return {
            display: 'block',
            position: 'absolute',
            left: `${x * scaleX}px`,
            top: `${y * scaleY}px`,
            width: `${w * scaleX}px`,
            height: `${h * scaleY}px`,
            border: `2px solid ${segmentColor}`,
            backgroundColor: `${segmentColor}25`,
            pointerEvents: 'none',
            boxSizing: 'border-box',
            zIndex: 10
        };
    }, [hoveredSegmentIndex, segments, imageLoaded]);

    const getLabelStyle = useCallback((): React.CSSProperties => {
        if (hoveredSegmentIndex === null || !imageRef.current || !imageLoaded) {
            return { display: 'none' };
        }

        const segment = segments.find(s => s.index === hoveredSegmentIndex);
        if (!segment || !segment.bbox) {
            return { display: 'none' };
        }

        const [x, y] = segment.bbox;
        const img = imageRef.current;

        const scaleX = img.clientWidth / img.naturalWidth;
        const scaleY = img.clientHeight / img.naturalHeight;

        const segmentColor = segment.color || '#22c55e';
        return {
            display: 'block',
            position: 'absolute',
            left: `${x * scaleX}px`,
            top: `${y * scaleY - 24}px`,
            backgroundColor: segmentColor,
            color: 'white',
            padding: '2px 8px',
            fontSize: '12px',
            fontWeight: 'bold',
            borderRadius: '4px 4px 0 0',
            zIndex: 11
        };
    }, [hoveredSegmentIndex, segments, imageLoaded]);

    const handleProcess = async () => {
        setProcessing(true);
        try {
            await ocrAPI.process(Number(id), {
                useHybrid: useHybrid,
                minConfidence: minConfidence
            });
            await fetchDocument();
        } catch (error: any) {
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
            const response = await ocrAPI.extract(Number(id), schemaPrompt);
            setToonOutput(response.data.toon_output);
            setExtractedFields(response.data.fields || []);
            setActiveTab('toon');
            await fetchDocument();
        } catch (error: any) {
            alert(error.response?.data?.detail || 'Extraction failed');
        } finally {
            setExtracting(false);
        }
    };

    const copyToClipboard = (text: string) => {
        navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    const handleGetEstimate = async () => {
        try {
            const res = await ocrAPI.getSegments(Number(id), { estimateOnly: true });
            setTokenEstimate(res.data.estimated_tokens);
        } catch (e) {
            console.error('Failed to get estimate:', e);
        }
    };

    const handleClassify = async () => {
        setClassifying(true);
        try {
            const res = await ocrAPI.getSegments(Number(id), { classify: true });
            setSegments(res.data.segments || []);
            setColorLegend(res.data.color_legend || {});
            setLayoutSummary(res.data.layout_summary || {});
            setClassified(true);
            setTokenEstimate(null);
        } catch (e: any) {
            alert(e.response?.data?.detail || 'Classification failed');
        } finally {
            setClassifying(false);
        }
    };

    const PipelineStep = ({ icon: Icon, title, status, details }: {
        icon: any;
        title: string;
        status: string;
        details?: string | null;
    }) => {
        const isCompleted = status === 'completed';
        const isPending = status === 'pending';

        return (
            <div className={`flex items-center gap-3 p-3 rounded-lg ${isCompleted ? 'bg-green-50' : isPending ? 'bg-gray-50' : 'bg-red-50'}`}>
                <div className={`w-8 h-8 rounded-full flex items-center justify-center ${isCompleted ? 'bg-green-100' : isPending ? 'bg-gray-200' : 'bg-red-100'}`}>
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

            <main className="flex-1 ml-[260px] p-6">
                {/* Header */}
                <div className="flex items-center gap-4 mb-6">
                    <button
                        onClick={() => router.push('/')}
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

                    {usage && usage.total_tokens > 0 && (
                        <div className="flex items-center gap-2 px-3 py-1.5 bg-yellow-500/10 rounded-lg">
                            <Coins className="w-4 h-4 text-yellow-500" />
                            <span className="text-yellow-500 text-sm font-medium">
                                {usage.total_tokens.toLocaleString()} tokens
                            </span>
                        </div>
                    )}
                </div>

                {/* Main Content Area */}
                {document.status === 'completed' && imageUrl ? (
                    <div className="grid grid-cols-2 gap-6">
                        {/* Left Panel - Document Image with Overlay */}
                        <div className="card-dark">
                            <div className="flex items-center gap-2 mb-4">
                                <Image className="w-4 h-4 text-gray-400" />
                                <h2 className="text-white font-medium">Document Preview</h2>
                            </div>
                            <div
                                ref={containerRef}
                                className="relative bg-gray-900 rounded-lg overflow-hidden"
                                style={{ maxHeight: '650px', overflow: 'auto' }}
                            >
                                <img
                                    ref={imageRef}
                                    src={imageUrl}
                                    alt="Document"
                                    className="w-full h-auto"
                                    onLoad={() => setImageLoaded(true)}
                                />
                                <div style={getHighlightStyle()}></div>
                                <div style={getLabelStyle()}>
                                    {hoveredSegmentIndex} - {segments.find(s => s.index === hoveredSegmentIndex)?.type || 'text'}
                                </div>
                            </div>
                            {hoveredSegmentIndex && (
                                <div className="mt-2 text-sm text-green-400">
                                    Highlighting Segment #{hoveredSegmentIndex}
                                </div>
                            )}
                        </div>

                        {/* Right Panel - Segments / OCR / TOON */}
                        <div className="space-y-4">
                            {/* Classification Controls */}
                            <div className="card-dark p-3">
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-sm font-medium text-white">Layout Classification</span>
                                    {classified && (
                                        <span className="text-xs text-green-400">✓ Classified</span>
                                    )}
                                </div>
                                {!classified ? (
                                    <div className="space-y-2">
                                        {tokenEstimate === null ? (
                                            <button
                                                onClick={handleGetEstimate}
                                                className="w-full py-2 bg-gray-700 text-white rounded-lg text-sm hover:bg-gray-600 transition"
                                            >
                                                📊 Get Token Estimate
                                            </button>
                                        ) : (
                                            <>
                                                <div className="p-2 bg-yellow-500/10 rounded-lg text-center">
                                                    <span className="text-yellow-400 text-sm">~{tokenEstimate.toLocaleString()} tokens</span>
                                                    <span className="text-gray-500 text-xs ml-2">for {segments.length} segments</span>
                                                </div>
                                                <button
                                                    onClick={handleClassify}
                                                    disabled={classifying}
                                                    className="w-full py-2 bg-purple-500 text-white rounded-lg text-sm font-medium flex items-center justify-center gap-2 hover:bg-purple-600 transition disabled:opacity-50"
                                                >
                                                    {classifying ? (
                                                        <><Loader2 className="w-4 h-4 animate-spin" /> Classifying...</>
                                                    ) : (
                                                        '🎨 Run Classification'
                                                    )}
                                                </button>
                                            </>
                                        )}
                                    </div>
                                ) : (
                                    <div className="flex flex-wrap gap-2">
                                        {Object.entries(layoutSummary).map(([type, count]) => (
                                            <span
                                                key={type}
                                                className="px-2 py-1 rounded text-xs font-medium text-white"
                                                style={{ backgroundColor: colorLegend[type] || '#6b7280' }}
                                            >
                                                {type}: {count}
                                            </span>
                                        ))}
                                    </div>
                                )}
                            </div>

                            {/* Tabs */}
                            <div className="flex gap-1 p-1 bg-[#141414] rounded-lg inline-flex">
                                <button
                                    onClick={() => setActiveTab('segments')}
                                    className={`px-4 py-2 rounded-md text-sm font-medium transition ${activeTab === 'segments' ? 'bg-[#1e1e1e] text-white' : 'text-gray-400 hover:text-white'}`}
                                >
                                    <List className="w-4 h-4 inline mr-2" />
                                    Segments ({segments.length})
                                </button>
                                <button
                                    onClick={() => setActiveTab('ocr')}
                                    className={`px-4 py-2 rounded-md text-sm font-medium transition ${activeTab === 'ocr' ? 'bg-[#1e1e1e] text-white' : 'text-gray-400 hover:text-white'}`}
                                >
                                    <Code className="w-4 h-4 inline mr-2" />
                                    OCR Text
                                </button>
                                <button
                                    onClick={() => setActiveTab('toon')}
                                    className={`px-4 py-2 rounded-md text-sm font-medium transition ${activeTab === 'toon' ? 'bg-[#1e1e1e] text-white' : 'text-gray-400 hover:text-white'}`}
                                >
                                    <Database className="w-4 h-4 inline mr-2" />
                                    TOON
                                </button>
                            </div>

                            {/* Tab Content */}
                            <div className="card-dark" style={{ maxHeight: '500px', overflow: 'auto' }}>
                                {activeTab === 'segments' ? (
                                    <div className="space-y-3">
                                        {segments.map((segment) => (
                                            <div
                                                key={segment.index}
                                                onMouseEnter={() => setHoveredSegmentIndex(segment.index)}
                                                onMouseLeave={() => setHoveredSegmentIndex(null)}
                                                className={`p-4 rounded-lg border transition cursor-pointer ${hoveredSegmentIndex === segment.index ? 'border-green-500 bg-green-500/10' : 'border-gray-700 hover:border-gray-500 bg-[#1a1a1a]'}`}
                                                style={{ borderLeftColor: segment.color || '#22c55e', borderLeftWidth: '4px' }}
                                            >
                                                <div className="flex items-center gap-2 mb-2">
                                                    <span className={`px-2 py-0.5 rounded text-xs font-bold ${hoveredSegmentIndex === segment.index ? 'bg-green-500 text-white' : 'bg-gray-600 text-gray-300'}`}>
                                                        {segment.index}
                                                    </span>
                                                    <span
                                                        className="px-1.5 py-0.5 rounded text-xs font-medium text-white capitalize"
                                                        style={{ backgroundColor: segment.color || '#6b7280' }}
                                                    >
                                                        {segment.type || 'text'}
                                                    </span>
                                                    <span className={`ml-auto text-xs ${segment.confidence >= 0.75 ? 'text-green-400' : 'text-yellow-400'}`}>
                                                        {(segment.confidence * 100).toFixed(0)}% conf
                                                    </span>
                                                </div>
                                                <p className="text-gray-200 text-sm whitespace-pre-wrap leading-relaxed">
                                                    {segment.text}
                                                </p>
                                            </div>
                                        ))}
                                    </div>
                                ) : activeTab === 'ocr' ? (
                                    <div className="relative">
                                        <button
                                            onClick={() => copyToClipboard(ocrText)}
                                            className="absolute top-2 right-2 p-2 bg-white/10 rounded-lg text-gray-400 hover:text-white transition"
                                        >
                                            {copied ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4" />}
                                        </button>
                                        <pre className="text-sm font-mono text-gray-300 whitespace-pre-wrap">
                                            {ocrText || 'No OCR text available'}
                                        </pre>
                                    </div>
                                ) : (
                                    <div className="space-y-3">
                                        {extractedFields.length > 0 ? (
                                            <>
                                                {extractedFields.map((field, idx) => (
                                                    <div
                                                        key={idx}
                                                        className="p-4 rounded-lg border border-gray-700 bg-[#1a1a1a]"
                                                        onMouseEnter={() => {
                                                            if (field.refs && field.refs.length > 0) {
                                                                setHoveredSegmentIndex(field.refs[0]);
                                                            }
                                                        }}
                                                        onMouseLeave={() => setHoveredSegmentIndex(null)}
                                                    >
                                                        <div className="flex items-center justify-between mb-2">
                                                            <span className="text-purple-400 font-medium text-sm">
                                                                {field.name}
                                                            </span>
                                                            {field.refs && field.refs.length > 0 && (
                                                                <div className="flex items-center gap-1">
                                                                    <span className="text-xs text-gray-500">from:</span>
                                                                    {field.refs.map((ref) => (
                                                                        <span
                                                                            key={ref}
                                                                            className="px-1.5 py-0.5 text-xs font-bold bg-green-500/20 text-green-400 rounded cursor-pointer hover:bg-green-500/30 transition"
                                                                            onMouseEnter={() => setHoveredSegmentIndex(ref)}
                                                                            onMouseLeave={() => setHoveredSegmentIndex(null)}
                                                                        >
                                                                            {ref}
                                                                        </span>
                                                                    ))}
                                                                </div>
                                                            )}
                                                        </div>
                                                        <p className="text-white text-sm">
                                                            {field.value || <span className="text-gray-500 italic">null</span>}
                                                        </p>
                                                    </div>
                                                ))}
                                                <details className="mt-4">
                                                    <summary className="text-gray-500 text-xs cursor-pointer hover:text-gray-400">
                                                        Show raw TOON output
                                                    </summary>
                                                    <pre className="mt-2 text-xs font-mono text-purple-300 whitespace-pre-wrap bg-black/30 p-2 rounded">
                                                        {toonOutput}
                                                    </pre>
                                                </details>
                                            </>
                                        ) : (
                                            <div className="text-center text-gray-500 py-8">
                                                <Database className="w-8 h-8 mx-auto mb-2 opacity-50" />
                                                <p>Run extraction to see structured fields</p>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>

                            {/* Schema Extraction Card */}
                            <div className="card-white">
                                <h2 className="font-medium text-gray-900 mb-3">Schema Extraction</h2>
                                <textarea
                                    value={schemaPrompt}
                                    onChange={(e) => setSchemaPrompt(e.target.value)}
                                    className="w-full border border-gray-200 rounded-lg p-2 text-gray-900 text-sm focus:outline-none focus:border-purple-500 resize-none"
                                    rows={2}
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
                        </div>
                    </div>
                ) : (
                    /* Pre-OCR Layout */
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                        <div className="space-y-6">
                            {usage?.pipeline && (
                                <div className="card-white">
                                    <h2 className="font-semibold text-gray-900 mb-4">Processing Pipeline</h2>
                                    <div className="space-y-2">
                                        <PipelineStep icon={Upload} title="Upload" status={usage.pipeline.upload.status} details={usage.pipeline.upload.filename} />
                                        <div className="flex justify-center"><div className="w-px h-4 bg-gray-200"></div></div>
                                        <PipelineStep icon={Zap} title="OCR Processing" status={usage.pipeline.ocr.status} details={usage.pipeline.ocr.confidence ? `${(usage.pipeline.ocr.confidence * 100).toFixed(0)}% • ${usage.pipeline.ocr.text_blocks} blocks` : undefined} />
                                        <div className="flex justify-center"><div className="w-px h-4 bg-gray-200"></div></div>
                                        <PipelineStep icon={Send} title="LLM Extraction" status={usage.pipeline.extraction.status} details={usage.pipeline.extraction.entities ? `${usage.pipeline.extraction.entities} entities • ${usage.pipeline.extraction.relations} relations` : undefined} />
                                    </div>
                                </div>
                            )}

                            {document.status === 'pending' && (
                                <div className="card-white">
                                    <h2 className="font-medium text-gray-900 mb-3">Run OCR</h2>
                                    <div className="flex items-center justify-between mb-4 p-3 bg-gray-50 rounded-lg">
                                        <div>
                                            <div className="text-sm font-medium text-gray-900">Hybrid OCR</div>
                                            <div className="text-xs text-gray-500">Use AI for low-confidence regions</div>
                                        </div>
                                        <button
                                            type="button"
                                            onClick={() => setUseHybrid(!useHybrid)}
                                            className={`relative w-11 h-6 rounded-full transition-colors ${useHybrid ? 'bg-blue-500' : 'bg-gray-300'}`}
                                        >
                                            <span className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform ${useHybrid ? 'translate-x-5' : 'translate-x-0'}`} />
                                        </button>
                                    </div>

                                    {useHybrid && (
                                        <>
                                            <div className="mb-4 p-3 bg-blue-50 rounded-lg">
                                                <div className="flex items-center justify-between mb-2">
                                                    <span className="text-sm font-medium text-gray-900">Confidence Threshold</span>
                                                    <span className="text-sm font-bold text-blue-600">{(minConfidence * 100).toFixed(0)}%</span>
                                                </div>
                                                <input
                                                    type="range"
                                                    min="0.5"
                                                    max="0.95"
                                                    step="0.05"
                                                    value={minConfidence}
                                                    onChange={(e) => setMinConfidence(parseFloat(e.target.value))}
                                                    className="w-full h-2 bg-blue-200 rounded-lg appearance-none cursor-pointer accent-blue-500"
                                                />
                                                <div className="flex justify-between text-xs text-gray-500 mt-1">
                                                    <span>50%</span>
                                                    <span>95%</span>
                                                </div>
                                                <div className="text-xs text-gray-600 mt-2 space-y-1">
                                                    <div className="flex items-start gap-1">
                                                        <span>💡</span>
                                                        <span><strong>Lower</strong> = more text reprocessed by AI (uses tokens, more accurate)</span>
                                                    </div>
                                                    <div className="flex items-start gap-1">
                                                        <span>💡</span>
                                                        <span><strong>Higher</strong> = less text sent to AI (saves tokens, keeps Tesseract)</span>
                                                    </div>
                                                </div>
                                            </div>
                                            <div className="mb-3 p-2 bg-yellow-50 border border-yellow-200 rounded-lg text-xs text-yellow-700">
                                                ⚠️ Hybrid mode uses AI tokens for low-confidence text
                                            </div>
                                        </>
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
                        </div>

                        <div className="lg:col-span-2 card-dark flex items-center justify-center" style={{ minHeight: '400px' }}>
                            <div className="text-center text-gray-500">
                                <FileText className="w-12 h-12 mx-auto mb-3 opacity-50" />
                                <p>Run OCR to see document analysis</p>
                            </div>
                        </div>
                    </div>
                )}
            </main>
        </div>
    );
}
