/**
 * Dashboard Component
 * Displays list of all cases with filtering and status
 */
import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { FileText, Clock, CheckCircle, XCircle, Trash2, Play } from 'lucide-react';
import { apiClient } from '@/api/client';
import { CaseListItem, CaseStatus } from '@/types';
import { formatDistance } from 'date-fns';

const statusIcons = {
  [CaseStatus.UPLOADED]: Clock,
  [CaseStatus.VALIDATING]: Clock,
  [CaseStatus.PREPROCESSING]: Clock,
  [CaseStatus.SEGMENTING]: Clock,
  [CaseStatus.CLASSIFYING]: Clock,
  [CaseStatus.COMPLETE]: CheckCircle,
  [CaseStatus.ERROR]: XCircle,
  [CaseStatus.CANCELLED]: XCircle,
};

const statusColors = {
  [CaseStatus.UPLOADED]: 'text-gray-500',
  [CaseStatus.VALIDATING]: 'text-blue-500',
  [CaseStatus.PREPROCESSING]: 'text-blue-500',
  [CaseStatus.SEGMENTING]: 'text-blue-500',
  [CaseStatus.CLASSIFYING]: 'text-blue-500',
  [CaseStatus.COMPLETE]: 'text-green-500',
  [CaseStatus.ERROR]: 'text-red-500',
  [CaseStatus.CANCELLED]: 'text-gray-400',
};

export const Dashboard = () => {
  const [cases, setCases] = useState<CaseListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<string>('all');
  const navigate = useNavigate();

  useEffect(() => {
    loadCases();
    // Refresh every 5 seconds if there are processing cases
    const interval = setInterval(() => {
      if (cases.some(c => c.status !== CaseStatus.COMPLETE && c.status !== CaseStatus.ERROR)) {
        loadCases();
      }
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const loadCases = async () => {
    try {
      const data = await apiClient.listCases({ limit: 100 });
      setCases(data);
      setError(null);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load cases');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (caseId: string) => {
    if (!confirm(`Delete case ${caseId}?`)) return;

    try {
      await apiClient.deleteCase(caseId);
      setCases(cases.filter(c => c.case_id !== caseId));
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Failed to delete case');
    }
  };

  const handleProcess = async (caseId: string) => {
    try {
      await apiClient.processCase(caseId);
      await loadCases();
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Failed to start processing');
    }
  };

  const filteredCases = filter === 'all'
    ? cases
    : cases.filter(c => c.status === filter);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto p-8">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Case Dashboard</h1>
          <p className="text-gray-600 mt-1">Manage and review stenosis detection cases</p>
        </div>
        <button
          onClick={() => navigate('/upload')}
          className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
        >
          Upload New Case
        </button>
      </div>

      {/* Filter Tabs */}
      <div className="flex space-x-2 mb-6 border-b border-gray-200">
        {[
          { key: 'all', label: 'All Cases' },
          { key: CaseStatus.COMPLETE, label: 'Complete' },
          { key: CaseStatus.PREPROCESSING, label: 'Processing' },
          { key: CaseStatus.ERROR, label: 'Errors' },
        ].map(tab => (
          <button
            key={tab.key}
            onClick={() => setFilter(tab.key)}
            className={`px-4 py-2 border-b-2 transition-colors ${
              filter === tab.key
                ? 'border-primary-600 text-primary-600 font-medium'
                : 'border-transparent text-gray-600 hover:text-gray-900'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
          {error}
        </div>
      )}

      {/* Cases Table */}
      <div className="bg-white shadow rounded-lg overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Case ID
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                File
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Upload Date
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Status
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Progress
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {filteredCases.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-6 py-12 text-center text-gray-500">
                  No cases found. Upload a new case to get started.
                </td>
              </tr>
            ) : (
              filteredCases.map(case_ => {
                const StatusIcon = statusIcons[case_.status];
                return (
                  <tr
                    key={case_.case_id}
                    className="hover:bg-gray-50 cursor-pointer"
                    onClick={() => navigate(`/case/${case_.case_id}`)}
                  >
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <FileText className="w-5 h-5 text-gray-400 mr-2" />
                        <span className="font-mono text-sm font-medium text-gray-900">
                          {case_.case_id}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <div className="text-sm text-gray-900">{case_.original_filename}</div>
                      {case_.dimensions && (
                        <div className="text-xs text-gray-500 font-mono">
                          {case_.dimensions.join(' × ')}
                        </div>
                      )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatDistance(new Date(case_.upload_date), new Date(), { addSuffix: true })}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <StatusIcon className={`w-5 h-5 ${statusColors[case_.status]} mr-2`} />
                        <span className="text-sm capitalize">{case_.status}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full transition-all ${
                            case_.status === CaseStatus.COMPLETE
                              ? 'bg-green-500'
                              : case_.status === CaseStatus.ERROR
                              ? 'bg-red-500'
                              : 'bg-blue-500'
                          }`}
                          style={{ width: `${case_.progress * 100}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-500 mt-1">
                        {(case_.progress * 100).toFixed(0)}%
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                      <div className="flex justify-end space-x-2" onClick={e => e.stopPropagation()}>
                        {case_.status === CaseStatus.UPLOADED && (
                          <button
                            onClick={() => handleProcess(case_.case_id)}
                            className="text-blue-600 hover:text-blue-900"
                            title="Start Processing"
                          >
                            <Play className="w-5 h-5" />
                          </button>
                        )}
                        <button
                          onClick={() => handleDelete(case_.case_id)}
                          className="text-red-600 hover:text-red-900"
                          title="Delete Case"
                        >
                          <Trash2 className="w-5 h-5" />
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
