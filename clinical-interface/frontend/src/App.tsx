/**
 * Main Application Component
 * Handles routing and global state
 */
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Dashboard } from '@/components/Dashboard';
import { UploadPage } from '@/components/UploadPage';
import { CaseViewer } from '@/components/CaseViewer';
import { Heart } from 'lucide-react';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="min-h-screen bg-gray-50">
          {/* Navigation Header */}
          <nav className="bg-white shadow-sm border-b border-gray-200">
            <div className="max-w-7xl mx-auto px-8 py-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <Heart className="w-8 h-8 text-red-500" />
                  <div>
                    <h1 className="text-xl font-bold text-gray-900">
                      Coronary Stenosis Detection System
                    </h1>
                    <p className="text-xs text-gray-500">
                      University of Haifa × Ziv Medical Center
                    </p>
                  </div>
                </div>
                <div className="flex items-center space-x-4">
                  <a
                    href="/dashboard"
                    className="text-gray-700 hover:text-gray-900 font-medium"
                  >
                    Dashboard
                  </a>
                  <a
                    href="/upload"
                    className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
                  >
                    Upload Case
                  </a>
                </div>
              </div>
            </div>
          </nav>

          {/* Main Content */}
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/case/:caseId" element={<CaseViewer />} />
          </Routes>

          {/* Footer */}
          <footer className="bg-white border-t border-gray-200 mt-12">
            <div className="max-w-7xl mx-auto px-8 py-6">
              <div className="text-center text-sm text-gray-600">
                <p className="mb-2">
                  <strong>Research Prototype</strong> - For research purposes only
                </p>
                <p className="text-xs text-gray-500">
                  This AI-assisted stenosis detection system is not FDA-approved for clinical diagnosis.
                  All results must be validated by a qualified radiologist or cardiologist.
                </p>
                <p className="text-xs text-gray-400 mt-3">
                  Supervised by Prof. Mario Boley | University of Haifa Information Systems Department
                </p>
              </div>
            </div>
          </footer>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
