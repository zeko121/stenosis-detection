/**
 * Upload Page Component
 * Handles file upload with drag-and-drop interface
 */
import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileUp, AlertCircle, CheckCircle } from 'lucide-react';
import { apiClient } from '@/api/client';
import type { UploadResponse } from '@/types';

export const UploadPage = () => {
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    setUploading(true);
    setError(null);
    setUploadResult(null);

    try {
      const result = await apiClient.uploadCase(file);
      setUploadResult(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Upload failed');
    } finally {
      setUploading(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/gzip': ['.nii.gz'],
      'application/nii': ['.nii'],
    },
    maxFiles: 1,
    disabled: uploading,
  });

  return (
    <div className="max-w-4xl mx-auto p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Upload CCTA Scan
        </h1>
        <p className="text-gray-600">
          Upload a cardiac CT angiography scan in NIFTI format for stenosis analysis
        </p>
      </div>

      {/* Upload Area */}
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-lg p-12 text-center cursor-pointer
          transition-colors duration-200
          ${isDragActive ? 'border-primary-500 bg-primary-50' : 'border-gray-300 hover:border-primary-400'}
          ${uploading ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        <input {...getInputProps()} />

        <div className="flex flex-col items-center space-y-4">
          {uploading ? (
            <>
              <Upload className="w-16 h-16 text-primary-500 animate-pulse" />
              <p className="text-lg font-medium text-gray-700">Uploading...</p>
            </>
          ) : (
            <>
              <FileUp className="w-16 h-16 text-gray-400" />
              <div>
                <p className="text-lg font-medium text-gray-700">
                  {isDragActive ? 'Drop the file here' : 'Drag & drop a CCTA scan here'}
                </p>
                <p className="text-sm text-gray-500 mt-1">or click to browse</p>
              </div>
              <p className="text-xs text-gray-400">
                Supported formats: .nii, .nii.gz (Max: 500MB)
              </p>
            </>
          )}
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start space-x-3">
          <AlertCircle className="w-5 h-5 text-red-500 mt-0.5" />
          <div>
            <p className="font-medium text-red-900">Upload Failed</p>
            <p className="text-sm text-red-700 mt-1">{error}</p>
          </div>
        </div>
      )}

      {/* Success Message */}
      {uploadResult && (
        <div className="mt-6 p-6 bg-green-50 border border-green-200 rounded-lg">
          <div className="flex items-start space-x-3 mb-4">
            <CheckCircle className="w-6 h-6 text-green-500 mt-0.5" />
            <div>
              <p className="font-medium text-green-900">Upload Successful!</p>
              <p className="text-sm text-green-700 mt-1">{uploadResult.message}</p>
            </div>
          </div>

          <div className="bg-white p-4 rounded-lg space-y-2">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-xs text-gray-500 uppercase">Case ID</p>
                <p className="font-mono font-medium text-gray-900">{uploadResult.case_id}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500 uppercase">Status</p>
                <p className="font-medium text-gray-900">{uploadResult.status}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500 uppercase">File Size</p>
                <p className="font-medium text-gray-900">
                  {uploadResult.upload_info.size_mb.toFixed(2)} MB
                </p>
              </div>
              <div>
                <p className="text-xs text-gray-500 uppercase">Dimensions</p>
                <p className="font-mono text-sm text-gray-900">
                  {uploadResult.upload_info.dimensions.join(' × ')}
                </p>
              </div>
            </div>

            <div className="mt-4 pt-4 border-t border-gray-200">
              <a
                href={`/case/${uploadResult.case_id}`}
                className="inline-block px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
              >
                View Case Details
              </a>
            </div>
          </div>
        </div>
      )}

      {/* Instructions */}
      <div className="mt-12 bg-blue-50 border border-blue-200 rounded-lg p-6">
        <h3 className="font-semibold text-blue-900 mb-3">Upload Guidelines</h3>
        <ul className="space-y-2 text-sm text-blue-800">
          <li className="flex items-start">
            <span className="mr-2">•</span>
            <span>Ensure the scan is in NIFTI format (.nii or .nii.gz)</span>
          </li>
          <li className="flex items-start">
            <span className="mr-2">•</span>
            <span>Cardiac CT Angiography (CCTA) scans are recommended</span>
          </li>
          <li className="flex items-start">
            <span className="mr-2">•</span>
            <span>Patient information should be anonymized before upload</span>
          </li>
          <li className="flex items-start">
            <span className="mr-2">•</span>
            <span>Processing typically takes 2-5 minutes depending on scan size</span>
          </li>
        </ul>
      </div>
    </div>
  );
};
