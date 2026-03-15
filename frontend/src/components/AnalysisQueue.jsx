export const AnalysisQueue = ({
  images,
  removeImage,
  predictSingleImage,
  predictAllImages,
  clearAllImages,
  allPredicted,
  isAnyPredicting
}) => {
  return (
    <div className="right-panel">
      {images.length > 0 ? (
        <section className="results-section">
          <div className="results-header">
            <h2 className="section-title">
              Analysis Queue [{images.length}]
            </h2>
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
              <button
                className="btn btn-predict-all"
                onClick={clearAllImages}
                disabled={isAnyPredicting}
                style={{ borderColor: 'var(--borders)', color: 'var(--text-dim)' }}
              >
                CLEAR ALL
              </button>
              {!allPredicted && images.length > 1 && (
                <button
                  className="btn btn-predict-all"
                  onClick={predictAllImages}
                  disabled={isAnyPredicting}
                >
                  RUN ALL
                </button>
              )}
            </div>
          </div>

          <div className="image-grid">
            {images.map(img => (
              <div key={img.id} className="image-card">
                <div className="image-preview-container">
                  <button className="remove-btn" onClick={() => removeImage(img.id)} aria-label="Remove image">
                    ✕
                  </button>
                  {img.isPredicting && <div className="scanning-line"></div>}
                  <img src={img.preview} alt={img.name} className="image-preview" />
                </div>

                <div className="card-content">
                  <p className="filename" title={img.name}>{img.name}</p>

                  {img.prediction === null && !img.isPredicting && (
                    <div className="result-badge pending">
                      Awaiting
                    </div>
                  )}

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

                  {img.prediction === 'PNEUMONIA' && (
                    <div className="result-badge pneumonia">
                      PNEUMONIA
                    </div>
                  )}

                  {img.prediction === 'NORMAL' && (
                    <div className="result-badge normal">
                      NORMAL
                    </div>
                  )}

                  {!img.prediction && (
                    <button
                      className="btn"
                      onClick={() => predictSingleImage(img.id)}
                      disabled={img.isPredicting || isAnyPredicting}
                    >
                      {img.isPredicting ? 'SCANNING...' : 'DIAGNOSE'}
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
