import { useState, useRef, useCallback } from 'react';
import { Header } from '../components/Header';
import { Footer } from '../components/Footer';
import { UploadPanel } from '../components/UploadPanel';
import { AnalysisQueue } from '../components/AnalysisQueue';
import '../index.css';

export const DiagnosticPage = () => {
  const [images, setImages] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);
  
  const allPredicted = images.length > 0 && images.every(img => img.prediction !== null);
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
      isPredicting: false,
    }));

    setImages(prev => [...prev, ...newImages]);
  };

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isDragging) setIsDragging(true);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFiles(e.dataTransfer.files);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      processFiles(e.target.files);
    }
  };

  const removeImage = (idToRemove) => {
    setImages(prev => {
      const imgToRemove = prev.find(img => img.id === idToRemove);
      if (imgToRemove) {
        URL.revokeObjectURL(imgToRemove.preview);
      }
      return prev.filter(img => img.id !== idToRemove);
    });
  };

  const clearAllImages = () => {
    setImages(prev => {
      prev.forEach(img => URL.revokeObjectURL(img.preview));
      return [];
    });
  };

  const predictSingleImage = async (id) => {
    const targetImage = images.find(img => img.id === id);
    if (!targetImage || targetImage.prediction) return;

    setImages(prev => prev.map(img => 
      img.id === id ? { ...img, isPredicting: true } : img
    ));

    const formData = new FormData();
    formData.append('files', targetImage.file);

    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Prediction request failed');
      }

      const data = await response.json();

      if (data && data.length > 0) {
        const result = data[0];
        setImages(prev => prev.map(img => 
          img.id === id ? { 
            ...img, 
            prediction: result.prediction,
            confidence: result.probability,
            isPredicting: false 
          } : img
        ));
      } else {
        throw new Error('No prediction returned');
      }
    } catch (error) {
      console.error('Error predicting image:', error);
      setImages(prev => prev.map(img => 
        img.id === id ? { 
          ...img, 
          isPredicting: false 
        } : img
      ));
    }
  };

  const predictAllImages = async () => {
    const unpredictedImages = images.filter(img => img.prediction === null);
    
    for (const img of unpredictedImages) {
      await predictSingleImage(img.id);
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
