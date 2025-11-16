/**
 * Case Viewer Component
 * Displays detailed case information, visualization, and stenosis analysis
 */
import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Download, AlertCircle } from 'lucide-react';
import { apiClient } from '@/api/client';
import { CaseDetail, CaseResults, CaseStatus, StenosisSeverity } from '@/types';

const severityColors = {
  [StenosisSeverity.NORMAL]: 'bg-green-100 text-green-800 border-green-300',
  [StenosisSeverity.MILD]: 'bg-yellow-100 text-yellow-800 border-yellow-300',
  [StenosisSeverity.MODERATE]: 'bg-orange-100 text-orange-800 border-orange-300',
  [StenosisSeverity.SEVERE]: 'bg-red-100 text-red-800 border-red-300',
};

const severityIndicators = {
  [StenosisSeverity.NORMAL]: '●○○○',
  [StenosisSeverity.MILD]: '●●○○',
  [StenosisSeverity.MODERATE]: '●●●○',
  [StenosisSeverity.SEVERE]: '●●●●',
};

export const CaseViewer = () => {
  const { caseId } = useParams<{ caseId: string }>();
  const navigate = useNavigate();
  const [caseDetail, setCaseDetail] = useState<CaseDetail | null>(null);
  const [results, setResults] = useState<CaseResults | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedPlane, setSelectedPlane] = useState<'axial' | 'coronal' | 'sagittal'>('axial');
  const [sliceIndex, setSliceIndex] = useState(0);
  const [showOverlay, setShowOverlay] = useState(true);

  useEffect(() => {
    if (!caseId) return;
    loadCaseData();

    // Poll for updates if processing
    const interval = setInterval(() => {
      if (caseDetail && caseDetail.status !== CaseStatus.COMPLETE && caseDetail.status !== CaseStatus.ERROR) {
        loadCaseData();
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [caseId]);

  const loadCaseData = async () => {
    if (!caseId) return;

    try {
      const detail = await apiClient.getCase(caseId);
      setCaseDetail(detail);

      if (detail.status === CaseStatus.COMPLETE) {
        const res = await apiClient.getCaseResults(caseId);
        setResults(res);
      }

      setError(null);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load case');
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadReport = async () => {
    if (!caseId) return;

    try {
      await apiClient.generateReport(caseId, { format: 'pdf', template: 'comprehensive' });
      // Download would happen automatically or open new window
      alert('Report generated! Check downloads.');
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Failed to generate report');
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  if (error || !caseDetail) {
    return (
      <div className="max-w-4xl mx-auto p-8">
        <div className="bg-red-50 border border-red-200 rounded-lg p-6">
          <AlertCircle className="w-6 h-6 text-red-500 mb-2" />
          <p className="text-red-900 font-medium">Error Loading Case</p>
          <p className="text-red-700 text-sm mt-1">{error}</p>
          <button
            onClick={() => navigate('/dashboard')}
            className="mt-4 text-primary-600 hover:text-primary-700"
          >
            Back to Dashboard
          </button>
        </div>
      </div>
    );
  }

  const isProcessing = caseDetail.status !== CaseStatus.COMPLETE && caseDetail.status !== CaseStatus.ERROR;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button
                onClick={() => navigate('/dashboard')}
                className="text-gray-600 hover:text-gray-900"
              >
                <ArrowLeft className="w-6 h-6" />
              </button>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Case {caseId}</h1>
                <p className="text-sm text-gray-600">{caseDetail.original_filename}</p>
              </div>
            </div>
            {caseDetail.status === CaseStatus.COMPLETE && (
              <button
                onClick={handleDownloadReport}
                className="flex items-center space-x-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
              >
                <Download className="w-5 h-5" />
                <span>Download Report</span>
              </button>
            )}
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto p-8">
        {/* Processing Status */}
        {isProcessing && (
          <div className="mb-6 bg-blue-50 border border-blue-200 rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <p className="font-medium text-blue-900">Processing Case...</p>
                <p className="text-sm text-blue-700 capitalize">{caseDetail.current_stage || 'Starting'}</p>
              </div>
              <div className="text-right">
                <p className="text-2xl font-bold text-blue-900">{(caseDetail.progress * 100).toFixed(0)}%</p>
              </div>
            </div>
            <div className="w-full bg-blue-200 rounded-full h-3">
              <div
                className="bg-blue-600 h-3 rounded-full transition-all duration-500"
                style={{ width: `${caseDetail.progress * 100}%` }}
              />
            </div>
          </div>
        )}

        {/* Error Status */}
        {caseDetail.status === CaseStatus.ERROR && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-6">
            <div className="flex items-start space-x-3">
              <AlertCircle className="w-6 h-6 text-red-500 mt-0.5" />
              <div>
                <p className="font-medium text-red-900">Processing Failed</p>
                <p className="text-sm text-red-700 mt-1">{caseDetail.error_message || 'Unknown error'}</p>
              </div>
            </div>
          </div>
        )}

        {/* Results Display */}
        {results && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Stenosis Analysis Cards */}
            <div className="lg:col-span-3">
              <h2 className="text-xl font-bold text-gray-900 mb-4">Stenosis Analysis</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {results.stenosis?.analyses.map(analysis => (
                  <div
                    key={analysis.vessel_name}
                    className={`p-6 rounded-lg border-2 ${severityColors[analysis.severity_class]}`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-bold text-lg">{analysis.vessel_name}</h3>
                      <span className="text-2xl">{severityIndicators[analysis.severity_class]}</span>
                    </div>
                    <div className="space-y-2">
                      <div>
                        <p className="text-sm opacity-75">Stenosis</p>
                        <p className="text-2xl font-bold">{(analysis.stenosis_percentage * 100).toFixed(1)}%</p>
                      </div>
                      <div>
                        <p className="text-sm opacity-75">Severity</p>
                        <p className="font-medium capitalize">{analysis.severity_class}</p>
                      </div>
                      <div>
                        <p className="text-sm opacity-75">Confidence</p>
                        <p className="font-medium">{(analysis.confidence_score * 100).toFixed(0)}%</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Summary Statistics */}
            {results.stenosis && (
              <div className="lg:col-span-3 bg-white rounded-lg shadow p-6">
                <h2 className="text-xl font-bold text-gray-900 mb-4">Summary</h2>
                <div className="grid grid-cols-3 gap-6">
                  <div>
                    <p className="text-sm text-gray-600">Most Severe Vessel</p>
                    <p className="text-2xl font-bold text-gray-900">{results.stenosis.most_severe_vessel}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Max Stenosis</p>
                    <p className="text-2xl font-bold text-gray-900">
                      {(results.stenosis.most_severe_stenosis * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Overall Severity</p>
                    <p className="text-2xl font-bold text-gray-900 capitalize">
                      {results.stenosis.overall_severity}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Segmentation Results */}
            {results.segmentation && (
              <div className="lg:col-span-3 bg-white rounded-lg shadow p-6">
                <h2 className="text-xl font-bold text-gray-900 mb-4">Vessel Segmentation</h2>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {results.segmentation.vessels.map(vessel => (
                    <div key={vessel.vessel_name} className="text-center p-4 bg-gray-50 rounded-lg">
                      <p className="font-medium text-gray-900">{vessel.vessel_name}</p>
                      <p className={`text-sm mt-1 ${vessel.detected ? 'text-green-600' : 'text-gray-400'}`}>
                        {vessel.detected ? '✓ Detected' : 'Not Detected'}
                      </p>
                    </div>
                  ))}
                </div>
                {results.segmentation.dice_score && (
                  <div className="mt-4 pt-4 border-t border-gray-200">
                    <p className="text-sm text-gray-600">
                      Dice Score: <span className="font-medium">{results.segmentation.dice_score.toFixed(3)}</span>
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Metadata */}
        <div className="mt-6 bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Case Metadata</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <p className="text-gray-600">Dimensions</p>
              <p className="font-mono">{caseDetail.dimensions?.join(' × ')}</p>
            </div>
            <div>
              <p className="text-gray-600">Spacing (mm)</p>
              <p className="font-mono">{caseDetail.spacing?.map(s => s.toFixed(2)).join(' × ')}</p>
            </div>
            <div>
              <p className="text-gray-600">File Size</p>
              <p>{caseDetail.file_size_mb?.toFixed(2)} MB</p>
            </div>
            <div>
              <p className="text-gray-600">Processing Time</p>
              <p>{caseDetail.processing_time_seconds?.toFixed(1)}s</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
