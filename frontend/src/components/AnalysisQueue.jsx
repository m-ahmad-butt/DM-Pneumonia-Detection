export const AnalysisQueue = ({
  images,
  removeImage,
  predictSingleImage,
  predictAllImages,
  clearAllImages,
  allPredicted,
  isAnyPredicting
}) => {

  // Summary counts for the header bar
  const predicted   = images.filter(img => img.prediction !== null);
  const pneumoniaCount = predicted.filter(img => img.prediction === 'PNEUMONIA').length;
  const normalCount    = predicted.filter(img => img.prediction === 'NORMAL').length;
  const errorCount     = images.filter(img => img.error !== null).length;

  return (
    <div className="right-panel">
      {images.length > 0 ? (
        <section className="results-section">
          <div className="results-header">
            <div>
              <h2 className="section-title">Analysis Queue [{images.length}]</h2>
              {predicted.length > 0 && (
                <p className="queue-summary">
                  {predicted.length}/{images.length} scanned
                  {pneumoniaCount > 0 && <span className="summary-pneumonia"> · {pneumoniaCount} Pneumonia</span>}
                  {normalCount    > 0 && <span className="summary-normal"> · {normalCount} Normal</span>}
                  {errorCount     > 0 && <span className="summary-error"> · {errorCount} Error</span>}
                </p>
              )}
            </div>
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
              <button
                className="btn btn-predict-all"
                onClick={clearAllImages}
                disabled={isAnyPredicting}
                style={{ borderColor: 'var(--borders)', color: 'var(--text-dim)' }}
              >
                CLEAR ALL
              </button>
              {!allPredicted && images.length >= 1 && (
                <button
                  className="btn btn-predict-all"
                  onClick={predictAllImages}
                  disabled={isAnyPredicting}
                >
                  {isAnyPredicting ? 'SCANNING...' : 'RUN ALL'}
                </button>
              )}
            </div>
          </div>

          <div className="image-grid">
            {images.map(img => (
              <div key={img.id} className={`image-card ${img.prediction === 'PNEUMONIA' ? 'card-pneumonia' : ''}`}>
                <div className="image-preview-container">
                  <button className="remove-btn" onClick={() => removeImage(img.id)} aria-label="Remove image">✕</button>
                  {img.isPredicting && <div className="scanning-line"></div>}
                  <img src={img.preview} alt={img.name} className="image-preview" />
                </div>

                <div className="card-content">
                  <p className="filename" title={img.name}>{img.name}</p>

                  {/* ── Awaiting ── */}
                  {img.prediction === null && !img.isPredicting && !img.error && (
                    <div className="result-badge pending">Awaiting</div>
                  )}

                  {/* ── Scanning ── */}
                  {img.isPredicting && (
                    <div className="result-badge pending" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      Scanning
                      <div className="loader-wrapper" style={{ marginLeft: '12px' }}>
                        <div className="pulse-dot"></div>
                        <div className="pulse-dot"></div>
                        <div className="pulse-dot"></div>
                      </div>
                    </div>
                  )}

                  {/* ── PNEUMONIA result ── */}
                  {img.prediction === 'PNEUMONIA' && (
                    <>
                      <div className="result-badge pneumonia">PNEUMONIA</div>
                      <p className="confidence-text confidence-pneumonia">
                        Confidence: {(img.confidence * 100).toFixed(1)}%
                      </p>
                    </>
                  )}

                  {/* ── NORMAL result ── */}
                  {img.prediction === 'NORMAL' && (
                    <>
                      <div className="result-badge normal">NORMAL</div>
                      <p className="confidence-text confidence-normal">
                        Confidence: {((1 - img.confidence) * 100).toFixed(1)}%
                      </p>
                    </>
                  )}

                  {/* ── Error ── */}
                  {img.error && !img.isPredicting && (
                    <div className="result-badge error" title={img.error}>ERROR</div>
                  )}

                  {/* ── Action button (only while not yet predicted) ── */}
                  {!img.prediction && !img.error && (
                    <button
                      className="btn"
                      onClick={() => predictSingleImage(img.id)}
                      disabled={img.isPredicting || isAnyPredicting}
                    >
                      {img.isPredicting ? 'SCANNING...' : 'DIAGNOSE'}
                    </button>
                  )}

                  {/* ── Retry button on error ── */}
                  {img.error && !img.isPredicting && (
                    <button
                      className="btn"
                      onClick={() => predictSingleImage(img.id)}
                      disabled={isAnyPredicting}
                      style={{ marginTop: '0.5rem' }}
                    >
                      RETRY
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </section>
      ) : (
        <div className="empty-state">
          <p>AWAITING SCANS...</p>
        </div>
      )}
    </div>
  );
};
