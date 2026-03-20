import { useState, useRef } from 'react';
import { Header } from '../components/Header';
import { Footer } from '../components/Footer';
import { UploadPanel } from '../components/UploadPanel';
import { AnalysisQueue } from '../components/AnalysisQueue';
import '../index.css';

// Reads from VITE_API_URL env var (set in Vercel dashboard).
// Falls back to localhost so local development works with no extra config.
const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
const API_URL  = `${API_BASE}/predict`;

export const DiagnosticPage = () => {
  const [images, setImages] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const allPredicted = images.length > 0 && images.every(img => img.prediction !== null || img.error !== null);
  const isAnyPredicting = images.some(img => img.isPredicting);

  const processFiles = (files) => {
    const validImageFiles = Array.from(files).filter(file => file.type.startsWith('image/'));
    if (validImageFiles.length === 0) return;

    const newImages = validImageFiles.map(file => ({
      id: Math.random().toString(36).substring(2, 9),
      file,
      preview: URL.createObjectURL(file),
      name: file.name,
      prediction: null,
      confidence: null,
      error: null,
      isPredicting: false,
    }));

    setImages(prev => [...prev, ...newImages]);
  };

  const handleDragEnter = (e) => { e.preventDefault(); e.stopPropagation(); setIsDragging(true); };
  const handleDragLeave = (e) => { e.preventDefault(); e.stopPropagation(); setIsDragging(false); };
  const handleDragOver  = (e) => { e.preventDefault(); e.stopPropagation(); if (!isDragging) setIsDragging(true); };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) processFiles(e.dataTransfer.files);
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files.length > 0) processFiles(e.target.files);
  };

  const removeImage = (idToRemove) => {
    setImages(prev => {
      const imgToRemove = prev.find(img => img.id === idToRemove);
      if (imgToRemove) URL.revokeObjectURL(imgToRemove.preview);
      return prev.filter(img => img.id !== idToRemove);
    });
  };

  const clearAllImages = () => {
    setImages(prev => { prev.forEach(img => URL.revokeObjectURL(img.preview)); return []; });
  };

  // ── Single image predict ──────────────────────────────────────────────────
  const predictSingleImage = async (id) => {
    const targetImage = images.find(img => img.id === id);
    if (!targetImage || targetImage.prediction || targetImage.error) return;

    setImages(prev => prev.map(img => img.id === id ? { ...img, isPredicting: true, error: null } : img));

    const formData = new FormData();
    formData.append('files', targetImage.file);

    try {
      const response = await fetch(API_URL, { method: 'POST', body: formData });
      if (!response.ok) throw new Error(`Server error ${response.status}`);

      const data = await response.json();
      if (data && data.length > 0) {
        const result = data[0];
        setImages(prev => prev.map(img =>
          img.id === id ? { ...img, prediction: result.prediction, confidence: result.probability, isPredicting: false } : img
        ));
      } else {
        throw new Error('No prediction returned');
      }
    } catch (error) {
      console.error('Prediction error:', error);
      setImages(prev => prev.map(img =>
        img.id === id ? { ...img, isPredicting: false, error: 'Request failed — is the backend running?' } : img
      ));
    }
  };

  // ── Batch predict ALL unpredicted images in a single API call ─────────────
  const predictAllImages = async () => {
    const unpredicted = images.filter(img => img.prediction === null && img.error === null && !img.isPredicting);
    if (unpredicted.length === 0) return;

    // Mark all as predicting
    setImages(prev => prev.map(img =>
      unpredicted.some(u => u.id === img.id) ? { ...img, isPredicting: true, error: null } : img
    ));

    const formData = new FormData();
    unpredicted.forEach(img => formData.append('files', img.file));

    try {
      const response = await fetch(API_URL, { method: 'POST', body: formData });
      if (!response.ok) throw new Error(`Server error ${response.status}`);

      const data = await response.json(); // array of { filename, prediction, probability }

      setImages(prev => prev.map(img => {
        const match = data.find(r => r.filename === img.name);
        if (!match) return img;
        return { ...img, prediction: match.prediction, confidence: match.probability, isPredicting: false };
      }));
    } catch (error) {
      console.error('Batch prediction error:', error);
      // Mark all as errored
      setImages(prev => prev.map(img =>
        unpredicted.some(u => u.id === img.id)
          ? { ...img, isPredicting: false, error: 'Request failed — is the backend running?' }
          : img
      ));
    }
  };

  return (
    <div className="app-container">
      <Header />
      <main className="main-content">
        <div className="layout-split">
          <UploadPanel
            isDragging={isDragging}
            handleDragEnter={handleDragEnter}
            handleDragLeave={handleDragLeave}
            handleDragOver={handleDragOver}
            handleDrop={handleDrop}
            fileInputRef={fileInputRef}
            handleFileInput={handleFileInput}
          />
          <AnalysisQueue
            images={images}
            removeImage={removeImage}
            predictSingleImage={predictSingleImage}
            predictAllImages={predictAllImages}
            clearAllImages={clearAllImages}
            allPredicted={allPredicted}
            isAnyPredicting={isAnyPredicting}
          />
        </div>
      </main>
      <Footer />
    </div>
  );
};
