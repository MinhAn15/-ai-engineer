'use client';

import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { userAPI } from '@/lib/api';

interface User {
    id: number;
    username: string;
    email?: string;
    is_admin: boolean;
}

interface AuthContextType {
    user: User | null;
    token: string | null;
    loading: boolean;
    isAuthenticated: boolean;
    login: (accessToken: string) => void;
    logout: () => void;
    refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
    const [user, setUser] = useState<User | null>(null);
    const [loading, setLoading] = useState(true);
    const [token, setToken] = useState<string | null>(null);

    useEffect(() => {
        // Only run on client
        const savedToken = localStorage.getItem('token');
        setToken(savedToken);

        if (savedToken) {
            fetchUser();
        } else {
            setLoading(false);
        }
    }, []);

    const fetchUser = async () => {
        try {
            const response = await userAPI.getMe();
            setUser(response.data);
        } catch (error) {
            console.error('Failed to fetch user:', error);
            logout();
        } finally {
            setLoading(false);
        }
    };

    const login = (accessToken: string) => {
        localStorage.setItem('token', accessToken);
        setToken(accessToken);
        fetchUser();
    };

    const logout = () => {
        localStorage.removeItem('token');
        setToken(null);
        setUser(null);
    };

    const value: AuthContextType = {
        user,
        token,
        loading,
        isAuthenticated: !!token,
        login,
        logout,
        refreshUser: fetchUser,
    };

    return (
        <AuthContext.Provider value={value}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error('useAuth must be used within AuthProvider');
    }
    return context;
}
