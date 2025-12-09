'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Upload, FolderOpen, Search, LogOut, Clock } from 'lucide-react';
import { useAuth } from '@/context/AuthContext';
import NeurondLogo from './NeurondLogo';

interface Document {
    id: number;
    filename: string;
}

export default function Sidebar({ recentDocs = [] }: { recentDocs?: Document[] }) {
    const { user, logout } = useAuth();
    const pathname = usePathname();

    const navItems = [
        { icon: Upload, label: 'New Upload', path: '/' },
        { icon: FolderOpen, label: 'My Documents', path: '/documents' },
        { icon: Search, label: 'Search', path: '/search' },
    ];

    return (
        <aside className="sidebar">
            {/* Logo */}
            <div className="flex items-center gap-3 p-4 mb-2">
                <NeurondLogo className="w-8 h-8" />
                <span className="font-semibold text-white text-sm">Document Intelligence</span>
            </div>

            {/* Navigation */}
            <nav className="flex-1 overflow-y-auto">
                {navItems.map((item) => (
                    <Link
                        key={item.path}
                        href={item.path}
                        className={`sidebar-nav-item ${pathname === item.path ? 'active' : ''}`}
                    >
                        <item.icon className="w-5 h-5" />
                        <span>{item.label}</span>
                    </Link>
                ))}

                {/* Recent section */}
                {recentDocs.length > 0 && (
                    <>
                        <div className="section-title mt-6">
                            <Clock className="w-4 h-4 inline mr-2" />
                            Recent
                        </div>
                        {recentDocs.slice(0, 5).map((doc) => (
                            <Link
                                key={doc.id}
                                href={`/documents/${doc.id}`}
                                className="history-item block"
                            >
                                {doc.filename}
                            </Link>
                        ))}
                    </>
                )}
            </nav>

            {/* User profile */}
            <div className="user-profile">
                <div className="user-avatar" style={{ background: '#3b82f6' }}>
                    {user?.username?.charAt(0).toUpperCase() || 'U'}
                </div>
                <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium text-white truncate">
                        {user?.username || 'User'}
                    </div>
                    <div className="text-xs text-gray-500 truncate">
                        {user?.email || ''}
                    </div>
                </div>
                <button
                    onClick={logout}
                    className="text-gray-500 hover:text-white transition p-1"
                    title="Logout"
                >
                    <LogOut className="w-4 h-4" />
                </button>
            </div>
        </aside>
    );
}
