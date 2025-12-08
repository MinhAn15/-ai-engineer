import { useState } from 'react';
import { NavLink } from 'react-router-dom';
import { Upload, FolderOpen, Search, LogOut, Clock } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import NeurondLogo from './NeurondLogo';

export default function Sidebar({ recentDocs = [] }) {
    const { user, logout } = useAuth();

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
                    <NavLink
                        key={item.path}
                        to={item.path}
                        className={({ isActive }) =>
                            `sidebar-nav-item ${isActive ? 'active' : ''}`
                        }
                    >
                        <item.icon className="w-5 h-5" />
                        <span>{item.label}</span>
                    </NavLink>
                ))}

                {/* Recent section */}
                {recentDocs.length > 0 && (
                    <>
                        <div className="section-title mt-6">
                            <Clock className="w-4 h-4 inline mr-2" />
                            Recent
                        </div>
                        {recentDocs.slice(0, 5).map((doc) => (
                            <NavLink
                                key={doc.id}
                                to={`/documents/${doc.id}`}
                                className="history-item block"
                            >
                                {doc.filename}
                            </NavLink>
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
